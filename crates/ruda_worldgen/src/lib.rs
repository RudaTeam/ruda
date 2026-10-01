//! Terrain generation: plains, hills and mountain ranges with caves and ore
//! deposits underground.
//!
//! The generator knows no blocks of its own: content packs pass in the blocks
//! it builds from through [`TerrainSettings`].

use noise::{Fbm, MultiFractal, NoiseFn, OpenSimplex};
use ruda_core::{BlockId, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, LocalPos};
use ruda_world::{Chunk, Generator};

/// What a terrain is made of.
#[derive(Clone, Debug)]
pub struct TerrainSettings {
    /// Land at or just above this height turns into sandy beaches.
    pub sea_level: i32,
    /// Top block of land away from beaches.
    pub grass: BlockId,
    /// The few layers under grass.
    pub dirt: BlockId,
    /// Top and filler of beaches.
    pub sand: BlockId,
    /// Everything deeper down.
    pub stone: BlockId,
    pub ores: Vec<Ore>,
    /// The bottom of the world: a single layer of `floor` at `min_y`, with
    /// nothing below it.
    pub floor: BlockId,
    pub min_y: i32,
}

/// An ore that replaces stone in small clusters within a band of heights.
#[derive(Clone, Debug)]
pub struct Ore {
    pub block: BlockId,
    pub min_y: i32,
    pub max_y: i32,
    /// Noise level, from -1 to 1, above which stone turns into this ore.
    /// Around 0.65 makes a common ore (a few percent of stone), 0.8 a rare
    /// one (under one percent).
    pub threshold: f64,
}

/// Generates chunks of a terrain, deterministically for a given seed.
#[derive(Debug)]
pub struct TerrainGenerator {
    settings: TerrainSettings,
    continents: Fbm<OpenSimplex>,
    hills: Fbm<OpenSimplex>,
    ridges: Fbm<OpenSimplex>,
    caves: Fbm<OpenSimplex>,
    ores: Vec<OpenSimplex>,
}

/// Caves never come closer to the surface than this, so the surface always
/// matches [`TerrainGenerator::surface_height`].
const CAVE_MIN_DEPTH: i32 = 5;
/// Cave noise is sampled every `CAVE_STEP` blocks and interpolated between.
const CAVE_STEP: usize = 4;
const CAVE_GRID: usize = CHUNK_SIZE as usize / CAVE_STEP + 1;
const CAVE_THRESHOLD: f64 = 0.38;
const ORE_FREQUENCY: f64 = 1.0 / 4.0;

impl TerrainGenerator {
    pub fn new(seed: u64, settings: TerrainSettings) -> Self {
        let mut seeds = Seeds(seed);
        Self {
            continents: Fbm::new(seeds.next())
                .set_octaves(4)
                .set_frequency(1.0 / 700.0),
            hills: Fbm::new(seeds.next())
                .set_octaves(4)
                .set_frequency(1.0 / 140.0),
            ridges: Fbm::new(seeds.next())
                .set_octaves(3)
                .set_frequency(1.0 / 320.0),
            caves: Fbm::new(seeds.next())
                .set_octaves(2)
                .set_frequency(1.0 / 56.0),
            ores: settings
                .ores
                .iter()
                .map(|_| OpenSimplex::new(seeds.next()))
                .collect(),
            settings,
        }
    }

    /// Height of the topmost block in the column at `(x, z)`. Cheap to call:
    /// it needs neither the chunk nor anything underground.
    pub fn surface_height(&self, x: i32, z: i32) -> i32 {
        let point = [f64::from(x), f64::from(z)];
        // -1 is deep lowland, 1 is high plateau.
        let continent = sample(&self.continents, point);
        let hills = sample(&self.hills, point);
        // Close to 1 along sharp crests.
        let ridge = 1.0 - sample(&self.ridges, point).abs();
        // Mountain ranges only rise from higher ground.
        let highland = ((continent + 0.2) / 1.2).clamp(0.0, 1.0);
        let height = f64::from(self.settings.sea_level)
            + 6.0
            + continent * 20.0
            + hills * 9.0
            + ridge.powi(3) * highland * highland * 70.0;
        height.floor() as i32
    }

    pub fn generate(&self, pos: ChunkPos) -> Chunk {
        let origin = pos.origin().0;
        let size = CHUNK_SIZE as usize;

        let mut heights = [[0; CHUNK_SIZE as usize]; CHUNK_SIZE as usize];
        for (z, row) in heights.iter_mut().enumerate() {
            for (x, height) in row.iter_mut().enumerate() {
                *height = self.surface_height(origin.x + x as i32, origin.z + z as i32);
            }
        }
        let highest = heights.iter().flatten().copied().max().unwrap_or(i32::MIN);
        if origin.y > highest || origin.y + CHUNK_SIZE - 1 < self.settings.min_y {
            return Chunk::filled(BlockId::AIR);
        }

        let caves = self.cave_grid(pos);
        let mut blocks = vec![BlockId::AIR; CHUNK_VOLUME];
        for (index, block) in blocks.iter_mut().enumerate() {
            let local = LocalPos::from_index(index);
            let (x, y, z) = (local.x() as usize, local.y() as usize, local.z() as usize);
            let world = origin + local.vec();
            if world.y <= self.settings.min_y {
                if world.y == self.settings.min_y {
                    *block = self.settings.floor;
                }
                continue;
            }
            let height = heights[z][x];
            let depth = height - world.y;
            if depth < 0 {
                continue;
            }
            let beach = height <= self.settings.sea_level + 1;
            *block = match depth {
                0 if beach => self.settings.sand,
                0 => self.settings.grass,
                1..=3 if beach => self.settings.sand,
                1..=3 => self.settings.dirt,
                _ if depth >= CAVE_MIN_DEPTH && sample_grid(&caves, x, y, z) > CAVE_THRESHOLD => {
                    BlockId::AIR
                }
                _ => self.ore_at(world.x, world.y, world.z),
            };
        }
        debug_assert_eq!(blocks.len(), size * size * size);
        Chunk::from_blocks(&blocks)
    }

    /// Cave noise at every `CAVE_STEP`th block of the chunk, edges included.
    fn cave_grid(&self, pos: ChunkPos) -> Vec<f64> {
        let origin = pos.origin().0;
        let mut grid = Vec::with_capacity(CAVE_GRID * CAVE_GRID * CAVE_GRID);
        for gy in 0..CAVE_GRID {
            for gz in 0..CAVE_GRID {
                for gx in 0..CAVE_GRID {
                    let [x, y, z] = [gx, gy, gz].map(|g| (g * CAVE_STEP) as i32);
                    // Squashing y stretches caves into wide, low tunnels.
                    grid.push(sample(
                        &self.caves,
                        [
                            f64::from(origin.x + x),
                            f64::from(origin.y + y) * 1.6,
                            f64::from(origin.z + z),
                        ],
                    ));
                }
            }
        }
        grid
    }

    /// Stone, or the first ore whose band and noise claim this block.
    fn ore_at(&self, x: i32, y: i32, z: i32) -> BlockId {
        let point = [x, y, z].map(|c| f64::from(c) * ORE_FREQUENCY);
        self.settings
            .ores
            .iter()
            .zip(&self.ores)
            .find(|(ore, noise)| {
                (ore.min_y..=ore.max_y).contains(&y) && sample(noise, point) > ore.threshold
            })
            .map_or(self.settings.stone, |(ore, _)| ore.block)
    }
}

impl Generator for TerrainGenerator {
    fn generate(&self, pos: ChunkPos) -> Chunk {
        TerrainGenerator::generate(self, pos)
    }

    fn surface_height(&self, x: i32, z: i32) -> Option<i32> {
        Some(TerrainGenerator::surface_height(self, x, z))
    }
}

/// Noise scaled to roughly -1..1: the library's generators stay within about
/// ±0.5.
fn sample<const N: usize>(noise: &impl NoiseFn<f64, N>, point: [f64; N]) -> f64 {
    noise.get(point) * 2.0
}

/// Trilinear interpolation of the coarse cave grid at a block of the chunk.
fn sample_grid(grid: &[f64], x: usize, y: usize, z: usize) -> f64 {
    let cell = |c: usize| (c / CAVE_STEP, (c % CAVE_STEP) as f64 / CAVE_STEP as f64);
    let ((gx, fx), (gy, fy), (gz, fz)) = (cell(x), cell(y), cell(z));
    let at = |x: usize, y: usize, z: usize| grid[(y * CAVE_GRID + z) * CAVE_GRID + x];
    let lerp = |a: f64, b: f64, t: f64| a + (b - a) * t;
    let plane = |y: usize| {
        let near = lerp(at(gx, y, gz), at(gx + 1, y, gz), fx);
        let far = lerp(at(gx, y, gz + 1), at(gx + 1, y, gz + 1), fx);
        lerp(near, far, fz)
    };
    lerp(plane(gy), plane(gy + 1), fy)
}

/// Derives independent noise seeds from the world seed (SplitMix64).
struct Seeds(u64);

impl Seeds {
    fn next(&mut self) -> u32 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        (z ^ (z >> 31)) as u32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruda_core::BlockPos;

    const GRASS: BlockId = BlockId::from_raw(10);
    const DIRT: BlockId = BlockId::from_raw(11);
    const SAND: BlockId = BlockId::from_raw(12);
    const STONE: BlockId = BlockId::from_raw(13);
    const ORE: BlockId = BlockId::from_raw(14);
    const FLOOR: BlockId = BlockId::from_raw(15);

    fn generator(seed: u64) -> TerrainGenerator {
        TerrainGenerator::new(
            seed,
            TerrainSettings {
                sea_level: 0,
                grass: GRASS,
                dirt: DIRT,
                sand: SAND,
                stone: STONE,
                ores: vec![Ore {
                    block: ORE,
                    min_y: -1000,
                    max_y: 1000,
                    threshold: 0.65,
                }],
                floor: FLOOR,
                min_y: -1024,
            },
        )
    }

    #[test]
    fn same_seed_same_world() {
        let pos = ChunkPos::new(3, 1, -2);
        assert_eq!(generator(7).generate(pos), generator(7).generate(pos));
        let heights = |seed| {
            let generator = generator(seed);
            (0..64)
                .map(|i| generator.surface_height(i * 37, i * 11))
                .collect::<Vec<_>>()
        };
        assert_ne!(heights(7), heights(8));
    }

    #[test]
    fn the_floor_is_one_layer_with_nothing_below() {
        let generator = generator(5);
        let mut blocks = vec![BlockId::AIR; CHUNK_VOLUME];
        generator
            .generate(ChunkPos::new(2, -32, 2))
            .copy_to(&mut blocks);
        for (index, &block) in blocks.iter().enumerate() {
            let y = LocalPos::from_index(index).y();
            assert_eq!(block == FLOOR, y == 0, "y = {y}");
        }
        assert_eq!(
            generator.generate(ChunkPos::new(2, -33, 2)).uniform(),
            Some(BlockId::AIR)
        );
    }

    #[test]
    fn sky_is_empty() {
        assert_eq!(
            generator(1).generate(ChunkPos::new(0, 40, 0)).uniform(),
            Some(BlockId::AIR)
        );
    }

    #[test]
    fn surface_matches_surface_height() {
        let generator = generator(42);
        for (x, z) in [(0, 0), (100, -250), (-777, 1234), (5000, 5000)] {
            let height = generator.surface_height(x, z);
            let top = BlockPos::new(x, height, z);
            let above = BlockPos::new(x, height + 1, z);
            let block = |pos: BlockPos| generator.generate(pos.chunk()).get(pos.local());
            assert!(matches!(block(top), GRASS | SAND), "({x}, {z})");
            assert_eq!(block(above), BlockId::AIR, "({x}, {z})");
        }
    }

    #[test]
    fn underground_is_mostly_stone_with_caves_and_ore() {
        let generator = generator(3);
        let mut blocks = vec![BlockId::AIR; CHUNK_VOLUME];
        let (mut stone, mut ore, mut air) = (0, 0, 0);
        for x in 0..4 {
            generator
                .generate(ChunkPos::new(x, -4, 0))
                .copy_to(&mut blocks);
            for &block in &blocks {
                match block {
                    STONE => stone += 1,
                    ORE => ore += 1,
                    BlockId::AIR => air += 1,
                    _ => {}
                }
            }
        }
        let total = f64::from(stone + ore + air);
        assert!(f64::from(stone) / total > 0.6, "stone {stone} of {total}");
        assert!(ore > 0 && air > 0, "ore {ore}, cave air {air} of {total}");
    }
}
