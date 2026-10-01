//! The far-away look of the world: beyond the chunks drawn in full, the
//! terrain is drawn from a coarse map of its surface.

use ruda_core::BlockId;

use crate::Generator;

/// Edge length of a cell, in blocks.
pub const LOD_CELL: i32 = 4;
/// Edge length of a tile, in cells.
pub const LOD_TILE_CELLS: usize = 64;
/// Edge length of a tile, in blocks.
pub const LOD_TILE_SIZE: i32 = LOD_CELL * LOD_TILE_CELLS as i32;

/// Position of a tile, counted in tiles: tile (1, 0) starts at block x 256.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LodTilePos {
    pub x: i32,
    pub z: i32,
}

impl LodTilePos {
    pub const fn new(x: i32, z: i32) -> Self {
        Self { x, z }
    }

    /// The tile that holds the column at block (x, z).
    pub fn containing(x: i32, z: i32) -> Self {
        Self::new(x.div_euclid(LOD_TILE_SIZE), z.div_euclid(LOD_TILE_SIZE))
    }

    /// Block x and z of the tile's minimum corner.
    pub fn origin(self) -> (i32, i32) {
        (self.x * LOD_TILE_SIZE, self.z * LOD_TILE_SIZE)
    }
}

/// The surface of a tile: for every cell, the height of its top block and
/// what that block is. Cells are in rows along x, one row per z.
#[derive(Clone, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct LodTile {
    pub heights: Vec<i16>,
    pub blocks: Vec<BlockId>,
}

impl LodTile {
    /// The tile as the generator would make it, sampled in the middle of
    /// every cell. `None` if the generator can't tell its surface.
    pub fn generate(generator: &dyn Generator, pos: LodTilePos) -> Option<Self> {
        let (x0, z0) = pos.origin();
        let cells = LOD_TILE_CELLS * LOD_TILE_CELLS;
        let mut heights = Vec::with_capacity(cells);
        let mut blocks = Vec::with_capacity(cells);
        for cz in 0..LOD_TILE_CELLS as i32 {
            for cx in 0..LOD_TILE_CELLS as i32 {
                let x = x0 + cx * LOD_CELL + LOD_CELL / 2;
                let z = z0 + cz * LOD_CELL + LOD_CELL / 2;
                let (height, block) = generator.surface(x, z)?;
                heights.push(height.clamp(i16::MIN.into(), i16::MAX.into()) as i16);
                blocks.push(block);
            }
        }
        Some(Self { heights, blocks })
    }

    /// Whether it has exactly one value per cell, as one from the network
    /// must before it is used.
    pub fn is_complete(&self) -> bool {
        let cells = LOD_TILE_CELLS * LOD_TILE_CELLS;
        self.heights.len() == cells && self.blocks.len() == cells
    }

    /// Height and top block of cell (x, z).
    pub fn cell(&self, x: usize, z: usize) -> (i16, BlockId) {
        let index = z * LOD_TILE_CELLS + x;
        (self.heights[index], self.blocks[index])
    }
}

#[cfg(test)]
mod tests {
    use ruda_core::ChunkPos;

    use super::*;
    use crate::Chunk;

    /// Height rises with x; the top is block 2 east of x = 0, 3 west of it.
    struct Slope;

    impl Generator for Slope {
        fn generate(&self, _: ChunkPos) -> Chunk {
            Chunk::filled(BlockId::AIR)
        }

        fn surface(&self, x: i32, _z: i32) -> Option<(i32, BlockId)> {
            Some((x / 8, BlockId::from_raw(if x >= 0 { 2 } else { 3 })))
        }
    }

    #[test]
    fn samples_the_middle_of_every_cell() {
        assert_eq!(LodTilePos::containing(-1, 300), LodTilePos::new(-1, 1));
        let tile = LodTile::generate(&Slope, LodTilePos::new(-1, 0)).unwrap();
        assert!(tile.is_complete());
        // The last cell of a row covers x −4 to −1; its middle is −2.
        assert_eq!(tile.cell(LOD_TILE_CELLS - 1, 0), (0, BlockId::from_raw(3)));
        // The first covers −256 to −253.
        assert_eq!(tile.cell(0, 5), (-254 / 8, BlockId::from_raw(3)));
        let east = LodTile::generate(&Slope, LodTilePos::new(0, 0)).unwrap();
        assert_eq!(east.cell(10, 0), (42 / 8, BlockId::from_raw(2)));
    }
}
