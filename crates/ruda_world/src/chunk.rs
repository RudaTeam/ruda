use ruda_core::{BlockId, CHUNK_VOLUME, Light, LocalPos};

use crate::Palette;

/// A 32×32×32 cube of blocks and the light at each of them.
///
/// Blocks are kept in a [`Palette`], so a chunk made of a single block takes
/// no per-block memory. Light changes all the time while it spreads, so it
/// is kept one value per block, except where it is the same everywhere.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    blocks: Palette<BlockId>,
    light: ChunkLight,
}

impl Chunk {
    /// A dark chunk where every block is `id`.
    pub fn filled(id: BlockId) -> Self {
        Self {
            blocks: Palette::filled(id),
            light: ChunkLight::uniform(Light::DARK),
        }
    }

    /// Builds a dark chunk from one id per block, in [`LocalPos`] index order.
    pub fn from_blocks(blocks: &[BlockId]) -> Self {
        Self {
            blocks: Palette::from_slice(blocks),
            light: ChunkLight::uniform(Light::DARK),
        }
    }

    pub fn get(&self, pos: LocalPos) -> BlockId {
        self.blocks.get(pos.index())
    }

    /// Sets a block and returns the one it replaced.
    pub fn set(&mut self, pos: LocalPos, id: BlockId) -> BlockId {
        self.blocks.set(pos.index(), id)
    }

    /// The block filling the whole chunk, if it is made of a single block.
    pub fn uniform(&self) -> Option<BlockId> {
        self.blocks.uniform()
    }

    /// Writes every block into `out`, in [`LocalPos`] index order.
    pub fn copy_to(&self, out: &mut [BlockId]) {
        self.blocks.copy_to(out);
    }

    /// The distinct blocks in the chunk; may include some that are gone.
    pub fn palette(&self) -> &[BlockId] {
        self.blocks.palette()
    }

    pub fn light(&self) -> &ChunkLight {
        &self.light
    }

    pub fn light_mut(&mut self) -> &mut ChunkLight {
        &mut self.light
    }

    pub fn set_light(&mut self, light: ChunkLight) {
        self.light = light;
    }
}

/// The light at every block of a chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkLight(LightStorage);

#[derive(Clone, Debug, PartialEq, Eq)]
enum LightStorage {
    Uniform(Light),
    /// One value per block, in [`LocalPos`] index order.
    Full(Box<[Light]>),
}

impl ChunkLight {
    /// The same light everywhere.
    pub fn uniform(light: Light) -> Self {
        Self(LightStorage::Uniform(light))
    }

    pub fn get(&self, pos: LocalPos) -> Light {
        match &self.0 {
            LightStorage::Uniform(light) => *light,
            LightStorage::Full(lights) => lights[pos.index()],
        }
    }

    pub fn set(&mut self, pos: LocalPos, light: Light) {
        match &mut self.0 {
            LightStorage::Uniform(current) if *current == light => {}
            _ => self.values_mut()[pos.index()] = light,
        }
    }

    /// The light, if it is the same everywhere.
    pub fn as_uniform(&self) -> Option<Light> {
        match &self.0 {
            LightStorage::Uniform(light) => Some(*light),
            LightStorage::Full(_) => None,
        }
    }

    /// Every value, in [`LocalPos`] index order, for changing many at once.
    pub fn values_mut(&mut self) -> &mut [Light] {
        if let LightStorage::Uniform(light) = self.0 {
            self.0 = LightStorage::Full(vec![light; CHUNK_VOLUME].into());
        }
        match &mut self.0 {
            LightStorage::Full(lights) => lights,
            LightStorage::Uniform(_) => unreachable!("just made full"),
        }
    }

    /// Writes every value into `out`, in [`LocalPos`] index order.
    pub fn copy_to(&self, out: &mut [Light]) {
        match &self.0 {
            LightStorage::Uniform(light) => out.fill(*light),
            LightStorage::Full(lights) => out.copy_from_slice(lights),
        }
    }

    /// Goes back to a single value if the light became the same everywhere.
    pub fn compact(&mut self) {
        if let LightStorage::Full(lights) = &self.0
            && let Some(&first) = lights.first()
            && lights.iter().all(|&light| light == first)
        {
            self.0 = LightStorage::Uniform(first);
        }
    }
}

/// Chunks travel as palettes: the blocks and the light each as their
/// distinct values plus packed indices. Incoming chunks are validated in
/// full, so a malformed one is rejected instead of panicking later.
#[cfg(feature = "serde")]
mod wire {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    use super::*;

    impl Serialize for ChunkLight {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            match &self.0 {
                LightStorage::Uniform(light) => Palette::filled(*light).serialize(serializer),
                LightStorage::Full(lights) => Palette::from_slice(lights).serialize(serializer),
            }
        }
    }

    impl<'de> Deserialize<'de> for ChunkLight {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let palette = Palette::<Light>::deserialize(deserializer)?;
            Ok(match palette.uniform() {
                Some(light) => Self::uniform(light),
                None => {
                    let mut lights = vec![Light::DARK; CHUNK_VOLUME];
                    palette.copy_to(&mut lights);
                    Self(LightStorage::Full(lights.into()))
                }
            })
        }
    }

    #[derive(Serialize, Deserialize)]
    struct ChunkWire {
        blocks: Palette<BlockId>,
        light: ChunkLight,
    }

    impl Serialize for Chunk {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            ChunkWire {
                blocks: self.blocks.clone(),
                light: self.light.clone(),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for Chunk {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            let wire = ChunkWire::deserialize(deserializer).map_err(D::Error::custom)?;
            Ok(Chunk {
                blocks: wire.blocks,
                light: wire.light,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(raw: u32) -> BlockId {
        BlockId::from_raw(raw)
    }

    #[test]
    fn keeps_blocks_and_light_apart() {
        let mut chunk = Chunk::filled(id(5));
        assert_eq!(chunk.set(LocalPos::new(1, 2, 3), id(7)), id(5));
        assert_eq!(chunk.get(LocalPos::new(1, 2, 3)), id(7));
        assert_eq!(chunk.light().as_uniform(), Some(Light::DARK));

        chunk
            .light_mut()
            .set(LocalPos::new(1, 2, 3), Light::rgb(4, 5, 6));
        assert_eq!(
            chunk.light().get(LocalPos::new(1, 2, 3)),
            Light::rgb(4, 5, 6)
        );
        assert_eq!(chunk.light().get(LocalPos::new(0, 0, 0)), Light::DARK);
        chunk.light_mut().set(LocalPos::new(1, 2, 3), Light::DARK);
        chunk.light_mut().compact();
        assert_eq!(chunk.light().as_uniform(), Some(Light::DARK));
    }

    #[cfg(feature = "serde")]
    #[test]
    fn serializes_and_rejects_malformed_chunks() {
        let mut chunk = Chunk::filled(id(1));
        for (n, pos) in LocalPos::all().step_by(13).enumerate() {
            chunk.set(pos, id(n as u32 % 5));
            chunk
                .light_mut()
                .set(pos, Light::new(n as u8 % 16, 0, 3, 0));
        }
        let bytes = postcard::to_allocvec(&chunk).unwrap();
        assert_eq!(postcard::from_bytes::<Chunk>(&bytes).unwrap(), chunk);

        let uniform = postcard::to_allocvec(&Chunk::filled(id(3))).unwrap();
        assert_eq!(
            postcard::from_bytes::<Chunk>(&uniform).unwrap(),
            Chunk::filled(id(3))
        );

        // Truncated index data must be rejected, not panic later.
        assert!(postcard::from_bytes::<Chunk>(&bytes[..bytes.len() / 2]).is_err());
    }
}
