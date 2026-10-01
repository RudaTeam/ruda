//! Clouds: soft clouds in a layer over the world, drifting with the wind,
//! parting around mountains and buildings, lit through the air like
//! everything else.
//!
//! A cloud isn't geometry. The shader works out how much cloud there is at
//! a point (`cloud_density` in `world.wgsl`) from three textures:
//! - patches of cloud, made from the sky's seed and slid along by the wind;
//! - finer, puffy detail for their edges, also from the seed;
//! - how high obstacles reach into the layer around the camera, so clouds
//!   thin out and part before they touch them.
//!
//! The patches depend only on the seed, the cover and how far the wind has
//! carried the air, so every player sees the same clouds without the server
//! sending them. Nothing is rebuilt as clouds move, so they can't pop in or
//! out.

use std::collections::HashMap;

use glam::{DVec2, DVec3, IVec2};
use rayon::prelude::*;
use ruda_core::{BlockId, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, LocalPos};
use ruda_world::Chunk;
use ruda_world::lod::{LOD_CELL, LOD_TILE_CELLS, LodTile, LodTilePos};

/// The bottom of the clouds: the tallest mountains rise through them.
pub const CLOUD_BOTTOM: f32 = 204.0;
/// The top of the tallest clouds.
pub const CLOUD_TOP: f32 = 244.0;
/// Blocky clouds, as in classic block games: cells of this many blocks a
/// side, in a layer this thick, around the middle of the soft clouds.
pub const BLOCKY_CELL: f32 = 12.0;
pub const BLOCKY_THICKNESS: f32 = 4.0;
pub const BLOCKY_BOTTOM: f32 = (CLOUD_BOTTOM + CLOUD_TOP - BLOCKY_THICKNESS) / 2.0;

/// Edge length of the patch texture, in texels, and how many blocks it
/// spans before it repeats.
pub(crate) const PATCH_SIZE: u32 = 512;
pub(crate) const PATCH_PERIOD: f64 = 4096.0;
/// The same for the detail texture, in all three directions.
pub(crate) const DETAIL_SIZE: u32 = 32;
pub(crate) const DETAIL_PERIOD: f64 = 96.0;

/// Edge length of the obstacle map around the camera, in cells of
/// `OBSTACLE_CELL` blocks: about 4 km across.
pub(crate) const OBSTACLE_SIZE: u32 = 1024;
pub(crate) const OBSTACLE_CELL: i32 = LOD_CELL;
/// Heights in the obstacle map start here; lower ones don't matter.
pub(crate) const OBSTACLE_BASE: i32 = CLOUD_BOTTOM as i32 - 32;
/// Clouds keep this many cells (of 4 blocks) clear of an obstacle's sides
/// before they start thinning out.
const CLEARANCE: usize = 1;
/// Then thin out over about twice this many cells.
const SOFTNESS: usize = 2;
/// While the world loads, obstacles change all the time: take them into
/// account at most this often, in seconds.
const OBSTACLE_UPDATES: f64 = 0.5;
/// The obstacle map moves in steps of this many cells as the camera does.
const OBSTACLE_STEP: i32 = 64;

/// What the clouds look like at a moment; the same for every player.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CloudSky {
    /// From the world's seed.
    pub seed: u64,
    /// How much of the sky clouds cover, 0 to 1.
    pub cover: f32,
    /// How far the wind has carried the air since the world began, in
    /// blocks along x and z; see [`ruda_core::WindMap::drift`].
    pub drift: DVec2,
}

/// The patch and detail textures for a seed.
pub(crate) struct CloudNoise {
    /// `PATCH_SIZE`² bytes. Every value is as common as every other, so
    /// the patches above 1 − cover cover exactly that much of the sky.
    pub patches: Vec<u8>,
    /// `DETAIL_SIZE`³ bytes: puffs, high in their middles.
    pub detail: Vec<u8>,
}

impl CloudNoise {
    pub(crate) fn new(seed: u64) -> Self {
        Self {
            patches: patches(seed),
            detail: detail(seed ^ 0x2545_f491_4f6c_dd1d),
        }
    }
}

/// Smooth, repeating noise: big patches with smaller ones on their edges.
fn patches(seed: u64) -> Vec<u8> {
    let size = PATCH_SIZE as usize;
    // Lattice spacing in texels (of 8 blocks; each divides the texture, so
    // it repeats seamlessly) and weight of each layer: clouds a hundred or
    // two blocks across, about as wide as the layer is tall.
    let octaves = [(32, 0.45), (16, 0.3), (8, 0.15), (4, 0.1)];
    let values: Vec<f32> = (0..size * size)
        .into_par_iter()
        .map(|i| {
            let (x, z) = ((i % size) as f32, (i / size) as f32);
            octaves
                .iter()
                .enumerate()
                .map(|(octave, &(spacing, weight))| {
                    let period = (size / spacing) as i32;
                    let seed =
                        seed.wrapping_add((octave as u64).wrapping_mul(0x9e37_79b9_7f4a_7c15));
                    weight * gradient_noise(seed, x / spacing as f32, z / spacing as f32, period)
                })
                .sum()
        })
        .collect();
    equalize(&values)
}

/// Turns values into their ranks: the result has every byte about equally
/// often, in the same order as the values.
fn equalize(values: &[f32]) -> Vec<u8> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_unstable_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut ranks = vec![0u8; values.len()];
    for (rank, &index) in order.iter().enumerate() {
        ranks[index] = (rank * 256 / values.len()) as u8;
    }
    ranks
}

/// Perlin's gradient noise at `(x, z)` on a lattice that repeats every
/// `period` cells, from −1 to 1 roughly.
fn gradient_noise(seed: u64, x: f32, z: f32, period: i32) -> f32 {
    let (x0, z0) = (x.floor(), z.floor());
    let (fx, fz) = (x - x0, z - z0);
    let corner = |dx: i32, dz: i32| {
        let cx = (x0 as i32 + dx).rem_euclid(period);
        let cz = (z0 as i32 + dz).rem_euclid(period);
        let angle = hash(seed, cx, cz, 0) * std::f32::consts::TAU;
        let (sin, cos) = angle.sin_cos();
        cos * (fx - dx as f32) + sin * (fz - dz as f32)
    };
    let fade = |t: f32| t * t * t * (t * (t * 6.0 - 15.0) + 10.0);
    let (u, v) = (fade(fx), fade(fz));
    let near = corner(0, 0) + (corner(1, 0) - corner(0, 0)) * u;
    let far = corner(0, 1) + (corner(1, 1) - corner(0, 1)) * u;
    (near + (far - near) * v) * std::f32::consts::SQRT_2
}

/// Puffy, repeating 3-D noise: one minus the distance to the nearest of
/// scattered points (Worley noise), in three sizes.
fn detail(seed: u64) -> Vec<u8> {
    let size = DETAIL_SIZE as usize;
    // Points per edge of the texture, and weight.
    let octaves = [(4, 0.6), (8, 0.27), (16, 0.13)];
    (0..size * size * size)
        .into_par_iter()
        .map(|i| {
            let p = [i % size, i / size % size, i / (size * size)].map(|c| c as f32 / size as f32);
            let value: f32 = octaves
                .iter()
                .enumerate()
                .map(|(octave, &(cells, weight))| {
                    let seed =
                        seed.wrapping_add((octave as u64).wrapping_mul(0xc2b2_ae3d_27d4_eb4f));
                    weight * (1.0 - worley(seed, p, cells))
                })
                .sum();
            (value.clamp(0.0, 1.0) * 255.0) as u8
        })
        .collect()
}

/// Distance from `p` (in the unit cube, repeating) to the nearest of one
/// random point per cell of a `cells`³ grid, in cell widths, at most 1.
fn worley(seed: u64, p: [f32; 3], cells: i32) -> f32 {
    let at = p.map(|c| c * cells as f32);
    let cell = at.map(|c| c.floor() as i32);
    let mut nearest = f32::MAX;
    for dz in -1..=1 {
        for dy in -1..=1 {
            for dx in -1..=1 {
                let c = [cell[0] + dx, cell[1] + dy, cell[2] + dz];
                let wrapped = c.map(|c| c.rem_euclid(cells));
                let key = (wrapped[2] * cells + wrapped[1]) * cells + wrapped[0];
                let point = [0, 1, 2].map(|axis| c[axis] as f32 + hash(seed, key, axis as i32, 1));
                let distance: f32 = (0..3).map(|axis| (point[axis] - at[axis]).powi(2)).sum();
                nearest = nearest.min(distance);
            }
        }
    }
    nearest.sqrt().min(1.0)
}

/// A number from 0 to 1 for every lattice point.
fn hash(seed: u64, x: i32, z: i32, salt: i32) -> f32 {
    let mut h = seed
        ^ u64::from(x as u32).wrapping_mul(0x9e37_79b9_7f4a_7c15)
        ^ u64::from(z as u32).wrapping_mul(0xc2b2_ae3d_27d4_eb4f)
        ^ u64::from(salt as u32).wrapping_mul(0x1656_67b1_9e37_79f9);
    h ^= h >> 31;
    h = h.wrapping_mul(0xbf58_476d_1ce4_e5b9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94d0_49bb_1331_11eb);
    h ^= h >> 31;
    (h >> 40) as f32 / (1u64 << 24) as f32
}

/// How high a chunk's solid blocks reach in each 4×4-block column: the top
/// of the highest one, or `None` if none of its columns gets near the clouds.
/// Entry `z * 8 + x` is column `(x, z)`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct ColumnTops([i16; 64]);

impl ColumnTops {
    const NONE: i16 = i16::MIN;
}

/// Where the solid blocks of a chunk reach up towards the clouds, if they do.
pub fn cloud_obstacles(
    pos: ChunkPos,
    chunk: &Chunk,
    solid: impl Fn(BlockId) -> bool,
) -> Option<ColumnTops> {
    let bottom = pos.origin().0.y;
    if bottom + CHUNK_SIZE <= OBSTACLE_BASE || chunk.palette().iter().all(|&block| !solid(block)) {
        return None;
    }
    let mut blocks = vec![BlockId::AIR; CHUNK_VOLUME];
    chunk.copy_to(&mut blocks);
    let cells = (CHUNK_SIZE / OBSTACLE_CELL) as usize;
    let mut tops = [ColumnTops::NONE; 64];
    for z in 0..CHUNK_SIZE {
        for x in 0..CHUNK_SIZE {
            let top = (0..CHUNK_SIZE).rev().find(|&y| {
                let local = LocalPos::new(x as u32, y as u32, z as u32);
                solid(blocks[local.index()])
            });
            if let Some(y) = top {
                let cell = (z / OBSTACLE_CELL) as usize * cells + (x / OBSTACLE_CELL) as usize;
                tops[cell] = tops[cell].max((bottom + y + 1) as i16);
            }
        }
    }
    tops.iter()
        .any(|&top| i32::from(top) > OBSTACLE_BASE)
        .then_some(ColumnTops(tops))
}

/// The same for far-away terrain: the surface of each cell, or `None` if
/// none gets near the clouds.
pub fn far_cloud_obstacles(tile: &LodTile) -> Option<Box<[i16]>> {
    let tops: Box<[i16]> = (0..LOD_TILE_CELLS * LOD_TILE_CELLS)
        .map(|i| tile.cell(i % LOD_TILE_CELLS, i / LOD_TILE_CELLS).0 + 1)
        .collect();
    tops.iter()
        .any(|&top| i32::from(top) > OBSTACLE_BASE)
        .then_some(tops)
}

/// How high obstacles reach into the clouds around the camera, as a texture.
#[derive(Debug, Default)]
pub(crate) struct ObstacleMap {
    chunks: HashMap<ChunkPos, ColumnTops>,
    tiles: HashMap<LodTilePos, Box<[i16]>>,
    changed: bool,
    built: Option<(IVec2, f64)>,
}

/// The obstacle map, ready for the GPU.
pub(crate) struct ObstacleImage {
    /// `OBSTACLE_SIZE`² bytes, a row after another: how high obstacles reach
    /// above `OBSTACLE_BASE`, widened and softened so clouds keep clear.
    pub texels: Vec<u8>,
    /// The cell (of `OBSTACLE_CELL` blocks) the first texel covers.
    pub origin: IVec2,
}

impl ObstacleMap {
    pub(crate) fn set_chunk(&mut self, pos: ChunkPos, tops: Option<ColumnTops>) {
        self.changed |= match tops {
            Some(tops) => self.chunks.insert(pos, tops) != Some(tops),
            None => self.chunks.remove(&pos).is_some(),
        };
    }

    pub(crate) fn set_tile(&mut self, pos: LodTilePos, tops: Option<Box<[i16]>>) {
        self.changed |= match tops {
            Some(tops) => self.tiles.insert(pos, tops.clone()).as_ref() != Some(&tops),
            None => self.tiles.remove(&pos).is_some(),
        };
    }

    /// A new map if the camera has moved far enough, or if obstacles have
    /// changed and the last map is old enough. `time` is in seconds.
    pub(crate) fn update(&mut self, camera: DVec3, time: f64) -> Option<ObstacleImage> {
        let cell = IVec2::new(
            (camera.x / f64::from(OBSTACLE_CELL)).floor() as i32,
            (camera.z / f64::from(OBSTACLE_CELL)).floor() as i32,
        );
        let half = OBSTACLE_SIZE as i32 / 2;
        // Moved only once the camera nears its edge.
        let near_edge =
            |origin: IVec2| (cell - origin - IVec2::splat(half)).abs().max_element() > half / 2;
        let (origin, due) = match self.built {
            Some((built, at)) if !near_edge(built) => {
                (built, self.changed && time - at >= OBSTACLE_UPDATES)
            }
            _ => (
                cell.div_euclid(IVec2::splat(OBSTACLE_STEP)) * OBSTACLE_STEP - IVec2::splat(half),
                true,
            ),
        };
        if !due {
            return None;
        }
        self.changed = false;
        self.built = Some((origin, time));
        Some(ObstacleImage {
            texels: self.build(origin),
            origin,
        })
    }

    fn build(&self, origin: IVec2) -> Vec<u8> {
        let size = OBSTACLE_SIZE as usize;
        let mut heights = vec![0u8; size * size];
        let mut raise = |cell: IVec2, top: i16| {
            let texel = cell - origin;
            if texel.min_element() >= 0 && texel.max_element() < size as i32 {
                let height = (i32::from(top) - OBSTACLE_BASE).clamp(0, 255) as u8;
                let at = &mut heights[texel.y as usize * size + texel.x as usize];
                *at = (*at).max(height);
            }
        };
        for (pos, tops) in &self.tiles {
            let (x, z) = pos.origin();
            let first = IVec2::new(x, z) / OBSTACLE_CELL;
            for (i, &top) in tops.iter().enumerate() {
                let offset = IVec2::new((i % LOD_TILE_CELLS) as i32, (i / LOD_TILE_CELLS) as i32);
                raise(first + offset, top);
            }
        }
        let cells = CHUNK_SIZE / OBSTACLE_CELL;
        for (pos, tops) in &self.chunks {
            let first = IVec2::new(pos.0.x, pos.0.z) * cells;
            for (i, &top) in tops.0.iter().enumerate() {
                if top != ColumnTops::NONE {
                    raise(first + IVec2::new(i as i32 % cells, i as i32 / cells), top);
                }
            }
        }
        // Clouds keep clear of the sides of obstacles, then thin out
        // towards them: the highest obstacle nearby, then smoothed.
        let filter = |image: &mut Vec<u8>, pass: &(dyn Fn(&[u8], &mut [u8]) + Sync)| {
            let mut out = vec![0u8; size * size];
            out.par_chunks_mut(size)
                .zip(image.par_chunks(size))
                .for_each(|(out, row)| pass(row, out));
            *image = out;
        };
        let widen = |row: &[u8], out: &mut [u8]| {
            for (x, out) in out.iter_mut().enumerate() {
                let range = x.saturating_sub(CLEARANCE)..(x + CLEARANCE + 1).min(size);
                *out = row[range].iter().copied().max().unwrap_or(0);
            }
        };
        let soften = |row: &[u8], out: &mut [u8]| {
            for (x, out) in out.iter_mut().enumerate() {
                let (sum, count) = (x as isize - SOFTNESS as isize..=(x + SOFTNESS) as isize)
                    .map(|x| row[x.clamp(0, size as isize - 1) as usize])
                    .fold((0u32, 0u32), |(sum, count), v| {
                        (sum + u32::from(v), count + 1)
                    });
                *out = (sum / count) as u8;
            }
        };
        filter(&mut heights, &widen);
        heights = transpose(&heights, size);
        filter(&mut heights, &widen);
        // Around that, a slope down, smoothed twice so it rounds off.
        let mut slope = heights.clone();
        filter(&mut slope, &soften);
        filter(&mut slope, &soften);
        slope = transpose(&slope, size);
        filter(&mut slope, &soften);
        filter(&mut slope, &soften);
        let heights = transpose(&heights, size);
        heights.iter().zip(slope).map(|(&h, s)| h.max(s)).collect()
    }
}

fn transpose(image: &[u8], size: usize) -> Vec<u8> {
    let mut out = vec![0u8; size * size];
    for (y, row) in image.chunks(size).enumerate() {
        for (x, &value) in row.iter().enumerate() {
            out[x * size + y] = value;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cover_sets_how_much_of_the_sky_is_cloud() {
        let noise = CloudNoise::new(42);
        for cover in [0.2f32, 0.35, 0.6] {
            let threshold = ((1.0 - cover) * 256.0) as u8;
            let clouds = noise.patches.iter().filter(|&&v| v >= threshold).count();
            let share = clouds as f32 / noise.patches.len() as f32;
            assert!((share - cover).abs() < 0.01, "cover {cover} gives {share}");
        }
    }

    #[test]
    fn noise_depends_only_on_the_seed() {
        let (a, b) = (CloudNoise::new(7), CloudNoise::new(7));
        assert_eq!(a.patches, b.patches);
        assert_eq!(a.detail, b.detail);
        assert_ne!(a.patches, CloudNoise::new(8).patches);
        assert_eq!(
            a.detail.len(),
            (DETAIL_SIZE * DETAIL_SIZE * DETAIL_SIZE) as usize
        );
        // Puffy: a fair spread of values.
        let (low, high) = a
            .detail
            .iter()
            .fold((255, 0), |(low, high), &v| (v.min(low), v.max(high)));
        assert!(low < 100 && high > 180, "{low} {high}");
    }

    #[test]
    fn the_patches_repeat_seamlessly() {
        // Opposite edges of the texture continue each other about as
        // smoothly as neighbouring rows inside it.
        let noise = CloudNoise::new(3);
        let size = PATCH_SIZE as usize;
        let row = |z: usize| &noise.patches[z * size..(z + 1) * size];
        let step = |a: &[u8], b: &[u8]| {
            a.iter()
                .zip(b)
                .map(|(&a, &b)| (i32::from(a) - i32::from(b)).abs())
                .sum::<i32>()
        };
        let inside = step(row(100), row(101));
        let across = step(row(size - 1), row(0));
        assert!(across < inside * 3, "{across} vs {inside}");
    }

    #[test]
    fn clouds_keep_clear_of_a_tower() {
        let mut map = ObstacleMap::default();
        // A tower to y = 250 over blocks 40..44 of chunk (1, 7, 1).
        let mut tops = [ColumnTops::NONE; 64];
        tops[2 * 8 + 2] = 250;
        map.set_chunk(ChunkPos::new(1, 7, 1), Some(ColumnTops(tops)));
        let image = map.update(DVec3::new(40.0, 120.0, 40.0), 0.0).unwrap();
        let at = |x: i32, z: i32| {
            let texel = IVec2::new(x, z) / OBSTACLE_CELL - image.origin;
            image.texels[texel.y as usize * OBSTACLE_SIZE as usize + texel.x as usize]
        };
        let top = (250 - OBSTACLE_BASE) as u8;
        // Over the tower and right next to it, the full height.
        assert_eq!(at(41, 41), top);
        assert_eq!(at(45, 41), top);
        // Further out, lower and lower, then nothing.
        assert!(at(53, 41) < at(45, 41) && at(53, 41) > top / 16);
        assert!(at(61, 41) < at(53, 41));
        assert_eq!(at(120, 41), 0);

        // Nothing changed: no new map until the camera nears the edge.
        assert!(map.update(DVec3::new(300.0, 120.0, 40.0), 10.0).is_none());
        assert!(map.update(DVec3::new(1800.0, 120.0, 40.0), 10.0).is_some());
        // Changes wait a moment.
        map.set_chunk(ChunkPos::new(1, 7, 1), None);
        assert!(map.update(DVec3::new(1800.0, 120.0, 40.0), 10.1).is_none());
        let cleared = map.update(DVec3::new(1800.0, 120.0, 40.0), 11.0).unwrap();
        assert!(cleared.texels.iter().all(|&v| v == 0));
    }
}
