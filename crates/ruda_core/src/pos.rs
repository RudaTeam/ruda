use glam::IVec3;

use crate::Face;

/// log2 of the chunk edge length.
pub const CHUNK_SHIFT: u32 = 5;
/// Chunks are cubes of 32×32×32 blocks.
pub const CHUNK_SIZE: i32 = 1 << CHUNK_SHIFT;
/// Number of blocks in a chunk.
pub const CHUNK_VOLUME: usize = 1 << (3 * CHUNK_SHIFT);

const LOCAL_MASK: i32 = CHUNK_SIZE - 1;

/// Position of a block in the world.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct BlockPos(pub IVec3);

impl BlockPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// The chunk this block belongs to.
    pub fn chunk(self) -> ChunkPos {
        ChunkPos(self.0.map(|c| c >> CHUNK_SHIFT))
    }

    /// Where this block sits inside its chunk.
    pub fn local(self) -> LocalPos {
        let IVec3 { x, y, z } = self.0.map(|c| c & LOCAL_MASK);
        LocalPos::new(x as u32, y as u32, z as u32)
    }

    /// The neighbouring block across `face`.
    pub fn offset(self, face: Face) -> Self {
        Self(self.0 + face.normal())
    }
}

/// Position of a chunk, counted in chunks: chunk (1, 0, 0) starts at block (32, 0, 0).
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct ChunkPos(pub IVec3);

impl ChunkPos {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self(IVec3::new(x, y, z))
    }

    /// The block at the chunk's minimum corner.
    pub fn origin(self) -> BlockPos {
        BlockPos(self.0 * CHUNK_SIZE)
    }

    pub fn block(self, local: LocalPos) -> BlockPos {
        BlockPos(self.origin().0 + local.vec())
    }

    /// The neighbouring chunk across `face`.
    pub fn offset(self, face: Face) -> Self {
        Self(self.0 + face.normal())
    }
}

/// Position of a block inside its chunk, packed as `y << 10 | z << 5 | x`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct LocalPos(u16);

impl LocalPos {
    /// Coordinates must be below [`CHUNK_SIZE`].
    pub fn new(x: u32, y: u32, z: u32) -> Self {
        debug_assert!(
            [x, y, z].iter().all(|&c| c < CHUNK_SIZE as u32),
            "local position ({x}, {y}, {z}) is outside the chunk"
        );
        Self((y << (2 * CHUNK_SHIFT) | z << CHUNK_SHIFT | x) as u16)
    }

    /// Index must be below [`CHUNK_VOLUME`].
    pub fn from_index(index: usize) -> Self {
        debug_assert!(index < CHUNK_VOLUME);
        Self(index as u16)
    }

    /// Every position in a chunk, in index order.
    pub fn all() -> impl Iterator<Item = LocalPos> {
        (0..CHUNK_VOLUME).map(Self::from_index)
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }

    pub fn x(self) -> u32 {
        u32::from(self.0) & LOCAL_MASK as u32
    }

    pub fn y(self) -> u32 {
        u32::from(self.0) >> (2 * CHUNK_SHIFT)
    }

    pub fn z(self) -> u32 {
        (u32::from(self.0) >> CHUNK_SHIFT) & LOCAL_MASK as u32
    }

    pub fn vec(self) -> IVec3 {
        IVec3::new(self.x() as i32, self.y() as i32, self.z() as i32)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn splits_negative_coordinates_into_chunk_and_local() {
        let pos = BlockPos::new(-1, 32, -33);
        assert_eq!(pos.chunk(), ChunkPos::new(-1, 1, -2));
        assert_eq!(pos.local().vec(), IVec3::new(31, 0, 31));
    }

    #[test]
    fn chunk_and_local_rebuild_the_block() {
        for pos in [
            BlockPos::new(0, 0, 0),
            BlockPos::new(-1, -1, -1),
            BlockPos::new(100, -70, 31),
            BlockPos::new(i32::MAX, i32::MIN, 12345),
        ] {
            assert_eq!(pos.chunk().block(pos.local()), pos);
        }
    }

    #[test]
    fn local_index_round_trips() {
        let local = LocalPos::new(3, 17, 31);
        assert_eq!((local.x(), local.y(), local.z()), (3, 17, 31));
        assert_eq!(LocalPos::from_index(local.index()), local);
        assert_eq!(LocalPos::all().count(), CHUNK_VOLUME);
    }
}
