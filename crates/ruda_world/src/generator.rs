use ruda_core::{BlockId, ChunkPos};

use crate::Chunk;

/// Produces the contents of chunks that have never been generated before.
pub trait Generator: Send + Sync {
    fn generate(&self, pos: ChunkPos) -> Chunk;

    /// Height of the topmost block of the column at `(x, z)`, if the
    /// generator can tell without generating chunks.
    fn surface_height(&self, x: i32, z: i32) -> Option<i32> {
        let _ = (x, z);
        None
    }

    /// Height and kind of the topmost block of the column at `(x, z)`, if
    /// the generator can tell cheaply. Far-away terrain is drawn from it.
    fn surface(&self, x: i32, z: i32) -> Option<(i32, BlockId)> {
        let _ = (x, z);
        None
    }
}
