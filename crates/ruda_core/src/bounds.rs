use crate::{BlockPos, CHUNK_SIZE, ChunkPos};

/// The vertical extent of a world: blocks exist from `min_y` to `max_y`,
/// both included.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct WorldBounds {
    pub min_y: i32,
    pub max_y: i32,
}

impl WorldBounds {
    /// From −128 to +255: 384 blocks, exactly 12 chunks tall. Sea level is
    /// at 0; the tallest mountains rise through the clouds at 221 to 227.
    pub const DEFAULT: Self = Self {
        min_y: -128,
        max_y: 255,
    };

    pub fn contains(self, pos: BlockPos) -> bool {
        (self.min_y..=self.max_y).contains(&pos.0.y)
    }

    /// Whether any block of the chunk is inside the world.
    pub fn contains_chunk(self, pos: ChunkPos) -> bool {
        let bottom = pos.origin().0.y;
        bottom + CHUNK_SIZE > self.min_y && bottom <= self.max_y
    }
}

impl Default for WorldBounds {
    fn default() -> Self {
        Self::DEFAULT
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_world_is_12_chunks_tall() {
        let bounds = WorldBounds::DEFAULT;
        let chunks = (-40..40)
            .filter(|&y| bounds.contains_chunk(ChunkPos::new(0, y, 0)))
            .count();
        assert_eq!(chunks, 12);
        assert!(bounds.contains(BlockPos::new(0, -128, 0)));
        assert!(!bounds.contains(BlockPos::new(0, -129, 0)));
        assert!(bounds.contains(BlockPos::new(0, 255, 0)));
        assert!(!bounds.contains(BlockPos::new(0, 256, 0)));
    }
}
