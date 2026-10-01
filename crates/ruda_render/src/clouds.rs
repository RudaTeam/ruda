//! Clouds: a layer of blocky cells drifting with the wind, slowly forming
//! and fading away, thinning out around mountains and tall buildings, and
//! shading the ground below.
//!
//! The pattern is fixed in "cloud space", which the wind slides over the
//! world, and changes slowly with time, so every player sees the same
//! clouds. How dense each cell is gets worked out on the worker pool for
//! moments [`KEYFRAME`] seconds apart; the renderer blends between the two
//! around the present, so cells fade in and out smoothly. A cell is only
//! added to or removed from the geometry while it can't be seen.

use std::collections::HashMap;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};

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
/// Seconds between the moments clouds are worked out for.
pub(crate) const KEYFRAME: f64 = 4.0;
/// Density at which a cell is fully opaque; below it, a cell is see-through,
/// more so the thinner it is.
pub(crate) const OPAQUE: f32 = 0.05;
/// Edge length of the texture that tells the terrain where clouds are.
pub(crate) const MASK_SIZE: u32 = 256;
/// Clouds reach at most this many cells from the camera.
const MAX_RADIUS: i32 = (MASK_SIZE as i32 - 1) / 2;
/// While the world loads, obstacles change all the time: take them into
/// account at most this often, in seconds.
const OBSTACLE_UPDATES: f64 = 0.5;
/// Obstacles are kept as cells of this many blocks.
const OBSTACLE_CELL: i32 = LOD_CELL;
/// Over how many cells clouds thin out towards an obstacle.
const CLEARANCE: f32 = 2.0;

/// What the clouds look like at a moment; the same for every player.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudSky {
    /// From the world's seed.
    pub seed: u64,
    /// How much of the sky clouds cover, 0 to 1.
    pub cover: f32,
    /// Seconds since the world began: how far the wind has moved them, and
    /// how they have changed.
    pub time: f64,
}

impl CloudSky {
    /// How far the wind has moved cloud space over the world, in blocks.
    pub fn drift(&self) -> DVec2 {
        drift(self.time)
    }
}

fn drift(time: f64) -> DVec2 {
    WIND * time
}

/// A layer of smooth noise: random values on a lattice `scale` cells apart,
/// and `period` seconds apart in time, blended in between.
struct Octave {
    seed: u64,
    scale: i32,
    period: f64,
    weight: f32,
}

/// Big blobs that change over a few minutes, and smaller detail that
/// changes faster.
const OCTAVES: [Octave; 2] = [
    Octave {
        seed: 0,
        scale: 7,
        period: 240.0,
        weight: 0.65,
    },
    Octave {
        seed: 0x5bd1_e995,
        scale: 3,
        period: 100.0,
        weight: 0.35,
    },
];

/// Eases a fraction from 0 to 1 in and out.
fn smooth(t: f32) -> f32 {
    t * t * (3.0 - 2.0 * t)
}

impl Octave {
    /// The lattice point at or before cell `x` along one axis, and how far
    /// past it the cell is, eased.
    fn locate(&self, x: i32) -> (i32, f32) {
        let past = x.rem_euclid(self.scale) as f32 / self.scale as f32;
        (x.div_euclid(self.scale), smooth(past))
    }

    /// The two slices of the lattice in time around `time`, as their seeds,
    /// and how to blend them: see [`through_time`].
    fn slices(&self, seed: u64, time: f64) -> ([u64; 2], f32, f32) {
        let slice = (time / self.period).floor();
        let t = smooth((time / self.period - slice) as f32);
        // Halfway between two unrelated slices, values stray less from the
        // middle; spread them out again, or the whole sky would have fewer
        // clouds every now and then.
        let spread = ((1.0 - t) * (1.0 - t) + t * t).sqrt();
        let seed =
            |slice: i64| seed ^ self.seed ^ (slice as u64).wrapping_mul(0xd6e8_feb8_6659_fd93);
        ([seed(slice as i64), seed(slice as i64 + 1)], t, spread)
    }
}

/// A lattice value blended between two slices in time.
fn through_time(now: f32, next: f32, t: f32, spread: f32) -> f32 {
    0.5 + (now + (next - now) * t - 0.5) / spread
}

/// Blends the values at the four lattice points around a cell.
fn plane([a, b, c, d]: [f32; 4], fx: f32, fz: f32) -> f32 {
    let near = a + (b - a) * fx;
    let far = c + (d - c) * fx;
    near + (far - near) * fz
}

/// Turns noise into density, so that about `cover` of the sky has cells at
/// least half opaque.
fn density(noise: f32, cover: f32) -> f32 {
    let half = 0.5 + (0.5 - cover.clamp(0.0, 1.0)) * 0.55;
    // Density grows twice as fast as the noise above where it starts.
    ((noise - half) * 2.0 + OPAQUE * 0.5).clamp(0.0, 1.0)
}

/// How dense cell (x, z) of cloud space is at `time`: 0 for clear sky, up
/// to 1. A cell is fully opaque from a small density on, and darker
/// underneath the denser it is; see [`cloud_opacity`].
pub fn cloud_density(seed: u64, cover: f32, x: i32, z: i32, time: f64) -> f32 {
    let noise = OCTAVES
        .iter()
        .map(|octave| {
            let ((lx, fx), (lz, fz)) = (octave.locate(x), octave.locate(z));
            let ([now, next], t, spread) = octave.slices(seed, time);
            let corners = [(0, 0), (1, 0), (0, 1), (1, 1)].map(|(dx, dz)| {
                let (x, z) = (lx + dx, lz + dz);
                through_time(hash(now, x, z), hash(next, x, z), t, spread)
            });
            octave.weight * plane(corners, fx, fz)
        })
        .sum();
    density(noise, cover)
}

/// How opaque a cell of a given density is, from 0 to 1.
pub fn cloud_opacity(density: f32) -> f32 {
    smooth((density / OPAQUE).clamp(0.0, 1.0))
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
    /// Box faces of the cells that stay fully opaque until the next
    /// keyframe, packed like this: word 0 is x and z of the first cell
    /// relative to `origin` (10 bits each) and the face (3 bits); word 1 is
    /// width and depth in cells, minus one (10 bits each).
    pub solid: Vec<[u32; 4]>,
    /// The same for the cells that can be seen through: clouds forming,
    /// fading away or thinning out near obstacles. They are drawn after the
    /// solid ones, so those show through them.
    pub faint: Vec<[u32; 4]>,
    /// Two bytes per cell, `MASK_SIZE` cells to a row: its density at the
    /// keyframe and at the next one, 0 to 255 for 0 to 1.
    pub density: Vec<u8>,
    /// Cloud-space cell of the first cell.
    pub origin: IVec2,
    /// Number of the keyframe: it is at `keyframe * KEYFRAME` seconds.
    pub keyframe: i64,
}

/// Where clouds may not go: per chunk, bit `z * 8 + x` set where blocks of
/// a 4×4-block column reach the clouds, and the same for far-away tiles, a
/// word per row of cells.
#[derive(Clone, Debug, Default)]
struct Obstacles {
    chunks: HashMap<ChunkPos, u64>,
    tiles: HashMap<LodTilePos, Box<[u64; LOD_TILE_CELLS]>>,
}

/// Keeps the clouds near the camera up to date, building them on the
/// worker pool.
#[derive(Debug)]
pub(crate) struct CloudField {
    /// Shared with the build in progress, copied if they change meanwhile.
    obstacles: Arc<Obstacles>,
    obstacles_changed: bool,
    /// What the latest build was started for.
    requested: Option<Request>,
    building: bool,
    done_tx: Sender<CloudMesh>,
    done_rx: Receiver<CloudMesh>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct Request {
    center: IVec2,
    radius: i32,
    seed: u64,
    cover: f32,
    keyframe: i64,
    time: f64,
}

/// Everything a build needs, to run on any thread.
struct CloudJob {
    origin: IVec2,
    size: usize,
    sky: CloudSky,
    keyframe: i64,
    obstacles: Arc<Obstacles>,
}

impl Default for CloudField {
    fn default() -> Self {
        let (done_tx, done_rx) = mpsc::channel();
        Self {
            obstacles: Arc::default(),
            obstacles_changed: false,
            requested: None,
            building: false,
            done_tx,
            done_rx,
        }
    }
}

impl CloudField {
    pub(crate) fn set_chunk(&mut self, pos: ChunkPos, columns: u64) {
        let known = self.obstacles.chunks.get(&pos).copied().unwrap_or(0);
        if known == columns {
            return;
        }
        let chunks = &mut Arc::make_mut(&mut self.obstacles).chunks;
        if columns == 0 {
            chunks.remove(&pos);
        } else {
            chunks.insert(pos, columns);
        }
        self.obstacles_changed = true;
    }

    pub(crate) fn set_tile(&mut self, pos: LodTilePos, rows: Option<[u64; LOD_TILE_CELLS]>) {
        let rows = rows.filter(|rows| rows.iter().any(|&row| row != 0));
        if self.obstacles.tiles.get(&pos).map(|rows| **rows) == rows {
            return;
        }
        let tiles = &mut Arc::make_mut(&mut self.obstacles).tiles;
        match rows {
            Some(rows) => tiles.insert(pos, Box::new(rows)),
            None => tiles.remove(&pos),
        };
        self.obstacles_changed = true;
    }

    /// Starts building the clouds again if the camera or the wind moved a
    /// cell, the next keyframe came, obstacles changed or the sky did, and
    /// returns the last build finished since the call before. With `wait`,
    /// builds right away instead, for pictures that must be complete.
    pub(crate) fn update(
        &mut self,
        camera: DVec3,
        sky: &CloudSky,
        reach: f32,
        wait: bool,
    ) -> Option<CloudMesh> {
        let mut finished = None;
        while let Ok(mesh) = self.done_rx.try_recv() {
            finished = Some(mesh);
            self.building = false;
        }
        if self.building {
            return finished;
        }
        match self.request(camera, sky, reach) {
            Some(job) if wait => Some(job.run()),
            Some(job) => {
                self.building = true;
                let done = self.done_tx.clone();
                rayon::spawn(move || {
                    let _ = done.send(job.run());
                });
                finished
            }
            None => finished,
        }
    }

    /// What to build, if the clouds need building again.
    fn request(&mut self, camera: DVec3, sky: &CloudSky, reach: f32) -> Option<CloudJob> {
        let drift = sky.drift();
        let cell = f64::from(CLOUD_CELL);
        let request = Request {
            center: IVec2::new(
                ((camera.x - drift.x) / cell).floor() as i32,
                ((camera.z - drift.y) / cell).floor() as i32,
            ),
            radius: ((reach / CLOUD_CELL as f32).ceil() as i32 + 1).min(MAX_RADIUS),
            seed: sky.seed,
            cover: sky.cover,
            keyframe: (sky.time / KEYFRAME).floor() as i64,
            time: sky.time,
        };
        if let Some(last) = self.requested {
            let obstacles = self.obstacles_changed && sky.time - last.time >= OBSTACLE_UPDATES;
            let same = Request {
                time: last.time,
                ..request
            } == last;
            if same && !obstacles {
                return None;
            }
        }
        self.obstacles_changed = false;
        self.requested = Some(request);
        Some(CloudJob {
            origin: request.center - IVec2::splat(request.radius),
            size: (2 * request.radius + 1) as usize,
            sky: *sky,
            keyframe: request.keyframe,
            obstacles: Arc::clone(&self.obstacles),
        })
    }
}

impl CloudJob {
    fn run(self) -> CloudMesh {
        let size = self.size;
        let times = [self.keyframe, self.keyframe + 1].map(|k| k as f64 * KEYFRAME);
        let mut densities = times.map(|time| densities(&self.sky, self.origin, size, time));

        // Clouds thin out towards anything that reaches them, and clear the
        // cells it reaches into while they drift from one keyframe to the
        // next.
        let clearance = clearance(&self.blocked(times.map(drift)), size);
        for densities in &mut densities {
            for (density, &clearance) in densities.iter_mut().zip(&clearance) {
                if clearance < 1.0 {
                    *density = density.min(clearance * OPAQUE);
                }
            }
        }

        let opaque = (OPAQUE * 255.0).ceil() as u8;
        let mut solid = vec![false; size * size];
        let mut faint = vec![false; size * size];
        let mut density = vec![0u8; (MASK_SIZE * MASK_SIZE * 2) as usize];
        for z in 0..size {
            for x in 0..size {
                let index = z * size + x;
                let [now, next] = densities
                    .each_ref()
                    .map(|d| (d[index] * 255.0).round() as u8);
                solid[index] = now >= opaque && next >= opaque;
                faint[index] = !solid[index] && (now > 0 || next > 0);
                let texel = (z * MASK_SIZE as usize + x) * 2;
                density[texel] = now;
                density[texel + 1] = next;
            }
        }
        CloudMesh {
            solid: mesh(&solid, size),
            faint: mesh(&faint, size),
            density,
            origin: self.origin,
            keyframe: self.keyframe,
        }
    }

    /// The cells that anything reaching the clouds is in, at some point
    /// while the wind moves them from one drift to the other.
    fn blocked(&self, [from, to]: [DVec2; 2]) -> Vec<bool> {
        let (origin, size) = (self.origin, self.size);
        let mut blocked = vec![false; size * size];
        let mut block = |block_x: i32, block_z: i32| {
            let start = DVec2::new(f64::from(block_x), f64::from(block_z));
            let end = start + f64::from(OBSTACLE_CELL);
            let min = (start - from).min(start - to);
            let max = (end - from).max(end - to);
            let first = (min / f64::from(CLOUD_CELL)).floor().as_ivec2() - origin;
            let last = ((max - 1e-6) / f64::from(CLOUD_CELL)).floor().as_ivec2() - origin;
            for z in first.y.max(0)..=last.y.min(size as i32 - 1) {
                for x in first.x.max(0)..=last.x.min(size as i32 - 1) {
                    blocked[z as usize * size + x as usize] = true;
                }
            }
        };
        for (pos, &columns) in &self.obstacles.chunks {
            for bit in Bits(columns) {
                let (x, z) = (bit % 8, bit / 8);
                block(
                    pos.0.x * CHUNK_SIZE + x as i32 * OBSTACLE_CELL,
                    pos.0.z * CHUNK_SIZE + z as i32 * OBSTACLE_CELL,
                );
            }
        }
        for (pos, rows) in &self.obstacles.tiles {
            let (tile_x, tile_z) = pos.origin();
            for (z, &row) in rows.iter().enumerate() {
                for x in Bits(row) {
                    block(
                        tile_x + x as i32 * OBSTACLE_CELL,
                        tile_z + z as i32 * OBSTACLE_CELL,
                    );
                }
            }
        }
        blocked
    }
}

/// The density of every cell of a square of cloud space at `time`, the same
/// as [`cloud_density`] gives, but working out each lattice point once.
fn densities(sky: &CloudSky, origin: IVec2, size: usize, time: f64) -> Vec<f32> {
    let mut noise = vec![0.0f32; size * size];
    for octave in &OCTAVES {
        // Lattice points, and where each column and row of cells is
        // between them.
        let columns: Vec<(i32, f32)> = (0..size as i32)
            .map(|x| octave.locate(origin.x + x))
            .collect();
        let rows: Vec<(i32, f32)> = (0..size as i32)
            .map(|z| octave.locate(origin.y + z))
            .collect();
        let (x0, z0) = (columns[0].0, rows[0].0);
        let width = (columns[size - 1].0 - x0 + 2) as usize;
        let depth = (rows[size - 1].0 - z0 + 2) as usize;
        let ([now, next], t, spread) = octave.slices(sky.seed, time);
        let mut lattice = Vec::with_capacity(width * depth);
        for z in 0..depth as i32 {
            for x in 0..width as i32 {
                let (x, z) = (x0 + x, z0 + z);
                lattice.push(through_time(hash(now, x, z), hash(next, x, z), t, spread));
            }
        }
        for (z, &(lz, fz)) in rows.iter().enumerate() {
            let row = (lz - z0) as usize * width;
            for (x, &(lx, fx)) in columns.iter().enumerate() {
                let at = row + (lx - x0) as usize;
                let corners = [
                    lattice[at],
                    lattice[at + 1],
                    lattice[at + width],
                    lattice[at + width + 1],
                ];
                noise[z * size + x] += octave.weight * plane(corners, fx, fz);
            }
        }
    }
    noise
        .into_iter()
        .map(|noise| density(noise, sky.cover))
        .collect()
}

/// How far clouds may thicken in each cell: 0 in blocked cells, rising to 1
/// over [`CLEARANCE`] cells away from them.
fn clearance(blocked: &[bool], size: usize) -> Vec<f32> {
    let mut clearance = vec![1.0f32; size * size];
    let reach = (CLEARANCE + 0.5).ceil() as i32;
    let size = size as i32;
    for (index, _) in blocked.iter().enumerate().filter(|(_, blocked)| **blocked) {
        let (x, z) = (index as i32 % size, index as i32 / size);
        for dz in -reach..=reach {
            for dx in -reach..=reach {
                let (nx, nz) = (x + dx, z + dz);
                if (0..size).contains(&nx) && (0..size).contains(&nz) {
                    let distance = ((dx * dx + dz * dz) as f32).sqrt();
                    let open = ((distance - 0.5) / CLEARANCE).clamp(0.0, 1.0);
                    let cell = &mut clearance[(nz * size + nx) as usize];
                    *cell = cell.min(open);
                }
            }
        }
    }
    clearance
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
            // Also halfway between two slices of the big blobs.
            for time in [0.0, 120.0, 1000.0] {
                let mut clouds = 0;
                for z in -200..200 {
                    for x in -200..200 {
                        let opacity = cloud_opacity(cloud_density(42, cover, x, z, time));
                        clouds += usize::from(opacity > 0.5);
                    }
                }
                let share = clouds as f32 / (400.0 * 400.0);
                assert!((share - cover).abs() < 0.08, "cover {cover} gives {share}");
            }
        }
    }

    #[test]
    fn a_square_at_once_matches_cell_by_cell() {
        let sky = CloudSky {
            seed: 9,
            cover: 0.5,
            time: 0.0,
        };
        let origin = IVec2::new(-20, 13);
        for time in [0.0, 37.5, 1e6] {
            let square = densities(&sky, origin, 40, time);
            for z in 0..40 {
                for x in 0..40 {
                    let cell = origin + IVec2::new(x, z);
                    let density = cloud_density(9, 0.5, cell.x, cell.y, time);
                    assert_eq!(square[(z * 40 + x) as usize], density);
                }
            }
        }
    }

    #[test]
    fn clouds_change_slowly() {
        let (mut most, mut changed) = (0.0f32, 0);
        for z in 0..100 {
            for x in 0..100 {
                let at = |time| cloud_density(5, 0.35, x, z, time);
                most = most.max((at(100.0 + KEYFRAME) - at(100.0)).abs());
                changed += usize::from(
                    (cloud_opacity(at(100.0)) > 0.5) != (cloud_opacity(at(700.0)) > 0.5),
                );
            }
        }
        // Even the fastest change is spread over a keyframe or more.
        assert!(most < 0.1, "from one keyframe to the next: {most}");
        assert!(
            changed > 1000,
            "in ten minutes, the sky is another: {changed}"
        );
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
    fn clouds_built_in_the_background_arrive() {
        let sky = CloudSky {
            seed: 3,
            cover: 0.5,
            time: 0.0,
        };
        let mut field = CloudField::default();
        assert!(field.update(DVec3::ZERO, &sky, 120.0, false).is_none());
        // Nothing new to build, but the build started arrives.
        let arrived = (0..1000).find_map(|_| {
            std::thread::sleep(std::time::Duration::from_millis(5));
            field.update(DVec3::ZERO, &sky, 120.0, false)
        });
        assert_eq!(arrived.map(|mesh| mesh.keyframe), Some(0));
    }

    #[test]
    fn clearance_grows_away_from_obstacles() {
        let mut blocked = vec![false; 81];
        blocked[40] = true;
        let clearance = clearance(&blocked, 9);
        assert_eq!(clearance[40], 0.0);
        assert_eq!(clearance[41], 0.25);
        assert_eq!(clearance[42], 0.75);
        assert_eq!(clearance[44], 1.0);
        // Diagonally, a step is longer.
        assert!(clearance[50] > clearance[41]);
    }

    #[test]
    fn clouds_thin_out_around_what_reaches_them() {
        let sky = CloudSky {
            seed: 7,
            cover: 1.0,
            time: 0.0,
        };
        let mut field = CloudField::default();
        let first = field.update(DVec3::ZERO, &sky, 120.0, true).unwrap();
        let density_at = |mesh: &CloudMesh, block_x: i32, block_z: i32| {
            let cell = IVec2::new(
                block_x.div_euclid(CLOUD_CELL),
                block_z.div_euclid(CLOUD_CELL),
            ) - mesh.origin;
            let texel = ((cell.y as u32 * MASK_SIZE + cell.x as u32) * 2) as usize;
            [mesh.density[texel], mesh.density[texel + 1]].map(|byte| f32::from(byte) / 255.0)
        };
        let solid = |mesh: &CloudMesh, block_x: i32, block_z: i32| {
            density_at(mesh, block_x, block_z)
                .iter()
                .all(|&density| cloud_opacity(density) > 0.99)
        };
        assert!(solid(&first, 40, 40) && solid(&first, 30, 40) && solid(&first, 100, 100));
        // Nothing changed: no new mesh.
        assert!(field.update(DVec3::ZERO, &sky, 120.0, true).is_none());

        // A peak at blocks 40..44 of chunk (1, 3, 1), x and z.
        field.set_chunk(ChunkPos::new(1, 3, 1), 1 << (2 * 8 + 2));
        assert!(
            field.update(DVec3::ZERO, &sky, 120.0, true).is_none(),
            "too soon"
        );
        let later = CloudSky { time: 1.0, ..sky };
        let parted = field.update(DVec3::ZERO, &later, 120.0, true).unwrap();
        assert_eq!(density_at(&parted, 40, 40), [0.0, 0.0]);
        // Next to it, the cloud is thin; farther away, as before.
        assert!(
            density_at(&parted, 28, 40)
                .iter()
                .all(|&density| cloud_opacity(density) < 0.5)
        );
        assert!(solid(&parted, 100, 100));
        // A new keyframe, a new mesh.
        let next = CloudSky {
            time: KEYFRAME,
            ..sky
        };
        assert_eq!(
            field
                .update(DVec3::ZERO, &next, 120.0, true)
                .unwrap()
                .keyframe,
            1
        );
    }
}
