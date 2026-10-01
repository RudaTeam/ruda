use std::collections::HashMap;

use ruda_core::{BlockId, BlockPos, ChunkPos};

use crate::Chunk;

/// The loaded part of a world: chunks by position.
#[derive(Clone, Debug, Default)]
pub struct World {
    chunks: HashMap<ChunkPos, Chunk>,
}

impl World {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks.get(&pos)
    }

    pub fn chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        self.chunks.get_mut(&pos)
    }

    /// Adds a chunk and returns the one it replaced.
    pub fn insert_chunk(&mut self, pos: ChunkPos, chunk: Chunk) -> Option<Chunk> {
        self.chunks.insert(pos, chunk)
    }

    pub fn remove_chunk(&mut self, pos: ChunkPos) -> Option<Chunk> {
        self.chunks.remove(&pos)
    }

    pub fn chunks(&self) -> impl Iterator<Item = (ChunkPos, &Chunk)> {
        self.chunks.iter().map(|(&pos, chunk)| (pos, chunk))
    }

    pub fn chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// The block at `pos`, or `None` if its chunk is not loaded.
    pub fn block(&self, pos: BlockPos) -> Option<BlockId> {
        Some(self.chunk(pos.chunk())?.get(pos.local()))
    }

    /// Sets a block and returns the one it replaced, or `None` (changing
    /// nothing) if its chunk is not loaded.
    pub fn set_block(&mut self, pos: BlockPos, id: BlockId) -> Option<BlockId> {
        Some(self.chunk_mut(pos.chunk())?.set(pos.local(), id))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocks_in_unloaded_chunks_are_none() {
        let mut world = World::new();
        let pos = BlockPos::new(-5, 40, 7);
        assert_eq!(world.block(pos), None);
        assert_eq!(world.set_block(pos, BlockId::UNKNOWN), None);

        world.insert_chunk(pos.chunk(), Chunk::filled(BlockId::AIR));
        assert_eq!(world.set_block(pos, BlockId::UNKNOWN), Some(BlockId::AIR));
        assert_eq!(world.block(pos), Some(BlockId::UNKNOWN));
    }
}
