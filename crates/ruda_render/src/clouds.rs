//! Clouds: a layer of blocky cells drifting with the wind, parting around
//! mountains and tall buildings, and shading the ground below.
//!
//! The pattern is fixed in "cloud space", which the wind slides over the
//! world, so every player sees the same clouds. Their geometry is rebuilt
//! only when the camera or the wind has moved a cell; in between, the
//! renderer just slides it along.

use std::collections::HashMap;

use glam::{DVec2, DVec3, IVec2};
use ruda_core::{BlockId, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, LocalPos};
use ruda_world::Chunk;
use ruda_world::lod::{LOD_CELL, LOD_TILE_CELLS, LodTile, LodTilePos};

/// Edge length of a cloud cell, in blocks.
pub const CLOUD_CELL: i32 = 12;
/// Height of the bottom of the clouds, about as high as the tallest
/// mountains.
pub const CLOUD_BOTTOM: i32 = 96;
pub const CLOUD_THICKNESS: i32 = 4;
/// Blocks per second the wind moves the clouds, mostly east.
pub const WIND: DVec2 = DVec2::new(1.0, 0.35);
/// Edge length of the texture that tells the terrain where clouds are.
pub(crate) const MASK_SIZE: u32 = 256;
/// Clouds reach at most this many cells from the camera.
const MAX_RADIUS: i32 = (MASK_SIZE as i32 - 1) / 2;
/// While the world loads, obstacles change all the time: take them into
/// account at most this often, in seconds.
const OBSTACLE_UPDATES: f64 = 0.5;
/// Obstacles are kept as cells of this many blocks.
const OBSTACLE_CELL: i32 = LOD_CELL;

/// What the clouds look like at a moment; the same for every player.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudSky {
    /// From the world's seed.
    pub seed: u64,
    /// How much of the sky clouds cover, 0 to 1.
    pub cover: f32,
    /// Seconds since the world began: how far the wind has moved them.
    pub time: f64,
}

impl CloudSky {
    /// How far the wind has moved cloud space over the world, in blocks.
    pub fn drift(&self) -> DVec2 {
        WIND * self.time
    }
}

/// Whether cell (x, z) of cloud space holds a cloud: two layers of smooth
/// noise, big blobs and smaller detail, over a threshold set by the cover.
pub fn cloud_at(seed: u64, cover: f32, x: i32, z: i32) -> bool {
    let noise = 0.65 * value_noise(seed, x, z, 7) + 0.35 * value_noise(seed ^ 0x5bd1_e995, x, z, 3);
    noise > 0.5 + (0.5 - cover.clamp(0.0, 1.0)) * 0.55
}

fn value_noise(seed: u64, x: i32, z: i32, scale: i32) -> f32 {
    let (lx, lz) = (x.div_euclid(scale), z.div_euclid(scale));
    let smooth = |t: f32| t * t * (3.0 - 2.0 * t);
    let fx = smooth(x.rem_euclid(scale) as f32 / scale as f32);
    let fz = smooth(z.rem_euclid(scale) as f32 / scale as f32);
    let at = |dx: i32, dz: i32| hash(seed, lx + dx, lz + dz);
    let near = at(0, 0) + (at(1, 0) - at(0, 0)) * fx;
    let far = at(0, 1) + (at(1, 1) - at(0, 1)) * fx;
    near + (far - near) * fz
}

/// A number from 0 to 1 for every lattice point.
fn hash(seed: u64, x: i32, z: i32) -> f32 {
    let mut h = seed
        ^ u64::from(x as u32).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(z as u32).wrapping_mul(0xc2b2_ae3d_27d4_eb4f);
    h ^= h >> 31;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// Where the solid blocks of a chunk reach the bottom of the clouds: bit
/// `z * 8 + x` is set for each 4×4-block column `(x, z)` with one.
pub fn cloud_obstacles(pos: ChunkPos, chunk: &Chunk, solid: impl Fn(BlockId) -> bool) -> u64 {
    let bottom = pos.origin().0.y;
    // Touching the bottom of a cloud counts.
    let from = CLOUD_BOTTOM - 1 - bottom;
    if from >= CHUNK_SIZE || chunk.palette().iter().all(|&block| !solid(block)) {
        return 0;
    }
    let mut blocks = vec![BlockId::AIR; CHUNK_VOLUME];
    chunk.copy_to(&mut blocks);
    let mut columns = 0u64;
    for y in from.max(0)..CHUNK_SIZE {
        for z in 0..CHUNK_SIZE {
            for x in 0..CHUNK_SIZE {
                let local = LocalPos::new(x as u32, y as u32, z as u32);
                if solid(blocks[local.index()]) {
                    columns |= 1 << ((z / OBSTACLE_CELL) * 8 + x / OBSTACLE_CELL);
                }
            }
        }
    }
    columns
}

/// The same for far-away terrain: a word per row of cells, bit `x` set
/// where the surface reaches the clouds.
pub fn far_cloud_obstacles(tile: &LodTile) -> [u64; LOD_TILE_CELLS] {
    let mut rows = [0u64; LOD_TILE_CELLS];
    for (z, row) in rows.iter_mut().enumerate() {
        for x in 0..LOD_TILE_CELLS {
            if i32::from(tile.cell(x, z).0) >= CLOUD_BOTTOM - 1 {
                *row |= 1 << x;
            }
        }
    }
    rows
}

/// The clouds near the camera, ready to draw.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct CloudMesh {
    /// Box faces, packed like this: word 0 is x and z of the first cell
    /// relative to `origin` (10 bits each) and the face (3 bits); word 1 is
    /// width and depth in cells, minus one (10 bits each).
    pub quads: Vec<[u32; 4]>,
    /// One byte per cell, 255 under cloud, `MASK_SIZE` cells to a row.
    pub mask: Vec<u8>,
    /// Cloud-space cell of the first cell.
    pub origin: IVec2,
}

/// Where clouds may not go, and when the mesh needs building again.
#[derive(Debug, Default)]
pub(crate) struct CloudField {
    /// Per chunk, bit `z * 8 + x` set where blocks of a 4×4-block column
    /// reach the clouds.
    chunks: HashMap<ChunkPos, u64>,
    /// The same for far-away tiles, a word per row of cells.
    tiles: HashMap<LodTilePos, Box<[u64; LOD_TILE_CELLS]>>,
    obstacles_changed: bool,
    built: Option<Built>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Built {
    center: IVec2,
    radius: i32,
    seed: u64,
    cover: f32,
    drift: DVec2,
    time: f64,
}

impl CloudField {
    pub(crate) fn set_chunk(&mut self, pos: ChunkPos, columns: u64) {
        let changed = if columns == 0 {
            self.chunks.remove(&pos).is_some()
        } else {
            self.chunks.insert(pos, columns) != Some(columns)
        };
        self.obstacles_changed |= changed;
    }

    pub(crate) fn set_tile(&mut self, pos: LodTilePos, rows: Option<[u64; LOD_TILE_CELLS]>) {
        let changed = match rows {
            Some(rows) if rows.iter().any(|&row| row != 0) => {
                self.tiles.insert(pos, Box::new(rows)).as_deref() != Some(&rows)
            }
            _ => self.tiles.remove(&pos).is_some(),
        };
        self.obstacles_changed |= changed;
    }

    /// A new mesh if the camera or the wind moved a cell since the last one,
    /// obstacles changed, or the sky did.
    pub(crate) fn update(
        &mut self,
        camera: DVec3,
        sky: &CloudSky,
        reach: f32,
    ) -> Option<CloudMesh> {
        let drift = sky.drift();
        let cell = f64::from(CLOUD_CELL);
        let center = IVec2::new(
            ((camera.x - drift.x) / cell).floor() as i32,
            ((camera.z - drift.y) / cell).floor() as i32,
        );
        let radius = ((reach / CLOUD_CELL as f32).ceil() as i32 + 1).min(MAX_RADIUS);
        let stale = match self.built {
            None => true,
            Some(built) => {
                built.center != center
                    || built.radius != radius
                    || built.seed != sky.seed
                    || built.cover != sky.cover
                    // Obstacles were kept clear only this far ahead.
                    || built.drift.distance(drift) >= cell * 0.9
                    || (self.obstacles_changed && sky.time - built.time >= OBSTACLE_UPDATES)
            }
        };
        if !stale {
            return None;
        }
        self.obstacles_changed = false;
        self.built = Some(Built {
            center,
            radius,
            seed: sky.seed,
            cover: sky.cover,
            drift,
            time: sky.time,
        });
        Some(self.build(center, radius, sky, drift))
    }

    fn build(&self, center: IVec2, radius: i32, sky: &CloudSky, drift: DVec2) -> CloudMesh {
        let size = (2 * radius + 1) as usize;
        let origin = center - IVec2::splat(radius);
        let mut cells = vec![false; size * size];
        for z in 0..size {
            for x in 0..size {
                cells[z * size + x] = cloud_at(
                    sky.seed,
                    sky.cover,
                    origin.x + x as i32,
                    origin.y + z as i32,
                );
            }
        }
        // Clear the cells that blocks reach into, now or before the wind has
        // carried the clouds another cell.
        let ahead = WIND.normalize_or_zero() * f64::from(CLOUD_CELL);
        let mut clear = |block_x: i32, block_z: i32| {
            let start = DVec2::new(f64::from(block_x), f64::from(block_z)) - drift;
            let end = start + f64::from(OBSTACLE_CELL);
            let min = start.min(start - ahead);
            let max = end.max(end - ahead);
            let first = (min / f64::from(CLOUD_CELL)).floor().as_ivec2() - origin;
            let last = ((max - 1e-6) / f64::from(CLOUD_CELL)).floor().as_ivec2() - origin;
            for z in first.y.max(0)..=last.y.min(size as i32 - 1) {
                for x in first.x.max(0)..=last.x.min(size as i32 - 1) {
                    cells[z as usize * size + x as usize] = false;
                }
            }
        };
        for (pos, &columns) in &self.chunks {
            for bit in Bits(columns) {
                let (x, z) = (bit % 8, bit / 8);
                clear(
                    pos.0.x * CHUNK_SIZE + x as i32 * OBSTACLE_CELL,
                    pos.0.z * CHUNK_SIZE + z as i32 * OBSTACLE_CELL,
                );
            }
        }
        for (pos, rows) in &self.tiles {
            let (tile_x, tile_z) = pos.origin();
            for (z, &row) in rows.iter().enumerate() {
                for x in Bits(row) {
                    clear(
                        tile_x + x as i32 * OBSTACLE_CELL,
                        tile_z + z as i32 * OBSTACLE_CELL,
                    );
                }
            }
        }

        let mut mask = vec![0u8; (MASK_SIZE * MASK_SIZE) as usize];
        for z in 0..size {
            for x in 0..size {
                if cells[z * size + x] {
                    mask[z * MASK_SIZE as usize + x] = 255;
                }
            }
        }
        CloudMesh {
            quads: mesh(&cells, size),
            mask,
            origin,
        }
    }
}

/// The set bits of a word, lowest first.
struct Bits(u64);

impl Iterator for Bits {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        (self.0 != 0).then(|| {
            let bit = self.0.trailing_zeros() as usize;
            self.0 &= self.0 - 1;
            bit
        })
    }
}

/// Faces 2 and 3 are the top and bottom (see `ruda_core::Face`); the others
/// the sides.
fn pack(x: usize, z: usize, width: usize, depth: usize, face: usize) -> [u32; 4] {
    [
        x as u32 | (z as u32) << 10 | (face as u32) << 20,
        (width as u32 - 1) | (depth as u32 - 1) << 10,
        0,
        0,
    ]
}

/// Tops and bottoms merged into rectangles, and the walls along the edges
/// of clouds merged into runs.
fn mesh(cells: &[bool], size: usize) -> Vec<[u32; 4]> {
    let at = |x: usize, z: usize| cells[z * size + x];
    let mut quads = Vec::new();
    let mut left = cells.to_vec();
    for z in 0..size {
        let mut x = 0;
        while x < size {
            if !left[z * size + x] {
                x += 1;
                continue;
            }
            let mut width = 1;
            while x + width < size && left[z * size + x + width] {
                width += 1;
            }
            let mut depth = 1;
            while z + depth < size && (x..x + width).all(|x| left[(z + depth) * size + x]) {
                depth += 1;
            }
            for row in z..z + depth {
                left[row * size + x..row * size + x + width].fill(false);
            }
            quads.push(pack(x, z, width, depth, 2));
            quads.push(pack(x, z, width, depth, 3));
            x += width;
        }
    }
    // Walls facing +x, −x along z, and +z, −z along x.
    for (face, dx, dz) in [(0, 1, 0), (1, -1, 0), (4, 0, 1), (5, 0, -1)] {
        for line in 0..size {
            let mut start = None;
            for step in 0..=size {
                let wall = step < size && {
                    let (x, z) = if dz == 0 { (line, step) } else { (step, line) };
                    let (nx, nz) = (x as i32 + dx, z as i32 + dz);
                    let open = !(0..size as i32).contains(&nx)
                        || !(0..size as i32).contains(&nz)
                        || !at(nx as usize, nz as usize);
                    at(x, z) && open
                };
                match (start, wall) {
                    (None, true) => start = Some(step),
                    (Some(first), false) => {
                        let length = step - first;
                        quads.push(if dz == 0 {
                            pack(line, first, 1, length, face)
                        } else {
                            pack(first, line, length, 1, face)
                        });
                        start = None;
                    }
                    _ => {}
                }
            }
        }
    }
    quads
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_sets_how_much_of_the_sky_is_cloud() {
        for cover in [0.2, 0.35, 0.6] {
            let mut clouds = 0;
            for z in -200..200 {
                for x in -200..200 {
                    clouds += usize::from(cloud_at(42, cover, x, z));
                }
            }
            let share = clouds as f32 / (400.0 * 400.0);
            assert!((share - cover).abs() < 0.08, "cover {cover} gives {share}");
        }
        assert!(!cloud_at(42, 0.0, 3, 4) || cloud_at(42, 1.0, 3, 4));
    }

    #[test]
    fn a_lone_cell_is_a_box_and_neighbours_share_their_tops() {
        assert_eq!(mesh(&[true], 1).len(), 6);
        // Two cells side by side: one top, one bottom, and four walls.
        let quads = mesh(&[true, true, false, false], 2);
        assert_eq!(quads.len(), 6);
        assert!(
            quads
                .iter()
                .any(|q| (q[0] >> 20) & 7 == 2 && q[1] & 1023 == 1)
        );
    }

    #[test]
    fn clouds_part_around_what_reaches_them() {
        let sky = CloudSky {
            seed: 7,
            cover: 1.0,
            time: 0.0,
        };
        let mut field = CloudField::default();
        let first = field.update(DVec3::ZERO, &sky, 120.0).unwrap();
        let cell_of = |mesh: &CloudMesh, block_x: i32, block_z: i32| {
            let cell = IVec2::new(
                block_x.div_euclid(CLOUD_CELL),
                block_z.div_euclid(CLOUD_CELL),
            ) - mesh.origin;
            mesh.mask[(cell.y as u32 * MASK_SIZE + cell.x as u32) as usize]
        };
        assert_eq!(cell_of(&first, 40, 40), 255);
        // Nothing changed: no new mesh.
        assert!(field.update(DVec3::ZERO, &sky, 120.0).is_none());

        // A peak at blocks 40..44 of chunk (1, 3, 1), x and z.
        field.set_chunk(ChunkPos::new(1, 3, 1), 1 << (2 * 8 + 2));
        assert!(field.update(DVec3::ZERO, &sky, 120.0).is_none(), "too soon");
        let later = CloudSky { time: 1.0, ..sky };
        let parted = field.update(DVec3::ZERO, &later, 120.0).unwrap();
        assert_eq!(cell_of(&parted, 40, 40), 0);
        // Upwind, where the wind is about to bring clouds, is clear too.
        assert_eq!(cell_of(&parted, 30, 40), 0);
        assert_eq!(cell_of(&parted, 80, 80), 255);
    }
}
