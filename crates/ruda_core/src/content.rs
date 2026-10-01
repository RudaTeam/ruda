use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::{BlockDef, BlockId, BlockRegistry, InvalidId, ResourceId};

/// Everything content packs registered for a session: blocks and the
/// textures they use. Built once with [`ContentBuilder`], immutable afterwards.
#[derive(Debug)]
pub struct Content {
    blocks: BlockRegistry,
    textures: BTreeMap<ResourceId, Cow<'static, [u8]>>,
    maps: BTreeMap<ResourceId, TextureMaps>,
}

/// How a texture's surface takes light, as two PNG images of the same size
/// as the texture, laid out as in labPBR 1.3.
///
/// - `normal`: red and green, the surface's tilt (the normal map's x and y,
///   with y pointing down the image); blue, how open each pixel is to light
///   (ambient occlusion, white fully open); alpha, height.
/// - `specular`: red, smoothness (white mirror-smooth); green, how much
///   light the surface reflects straight back (0 to 229 for 0% to 90%) or,
///   from 230, that it is a metal; blue, porosity; alpha, how much it glows
///   (0 to 254), 255 for not at all.
#[derive(Clone, Debug)]
pub struct TextureMaps {
    pub normal: Cow<'static, [u8]>,
    pub specular: Cow<'static, [u8]>,
}

impl Content {
    pub fn blocks(&self) -> &BlockRegistry {
        &self.blocks
    }

    /// PNG-encoded texture, if one is registered under `id`.
    pub fn texture(&self, id: &ResourceId) -> Option<&[u8]> {
        self.textures.get(id).map(|png| &**png)
    }

    /// All textures, sorted by id.
    pub fn textures(&self) -> impl Iterator<Item = (&ResourceId, &[u8])> {
        self.textures.iter().map(|(id, png)| (id, &**png))
    }

    /// How the texture `id` takes light, if it says.
    pub fn texture_maps(&self, id: &ResourceId) -> Option<&TextureMaps> {
        self.maps.get(id)
    }
}

/// Collects content from the engine and content packs at startup.
#[derive(Debug)]
pub struct ContentBuilder {
    blocks: BlockRegistry,
    textures: BTreeMap<ResourceId, Cow<'static, [u8]>>,
    maps: BTreeMap<ResourceId, TextureMaps>,
}

impl ContentBuilder {
    pub fn new() -> Self {
        Self {
            blocks: BlockRegistry::with_engine_blocks(),
            textures: BTreeMap::new(),
            maps: BTreeMap::new(),
        }
    }

    pub fn add_block(&mut self, def: BlockDef) -> Result<BlockId, ContentError> {
        self.blocks.register(def)
    }

    /// Registers a PNG texture. It is only decoded by clients that draw it.
    pub fn add_texture(
        &mut self,
        id: ResourceId,
        png: impl Into<Cow<'static, [u8]>>,
    ) -> Result<(), ContentError> {
        if self.textures.contains_key(&id) {
            return Err(ContentError::DuplicateTexture(id));
        }
        self.textures.insert(id, png.into());
        Ok(())
    }

    /// Says how a registered texture takes light; see [`TextureMaps`].
    /// Textures without maps are matte and flat.
    pub fn add_texture_maps(
        &mut self,
        id: ResourceId,
        maps: TextureMaps,
    ) -> Result<(), ContentError> {
        if !self.textures.contains_key(&id) {
            return Err(ContentError::MissingTexture(id));
        }
        if self.maps.contains_key(&id) {
            return Err(ContentError::DuplicateTexture(id));
        }
        self.maps.insert(id, maps);
        Ok(())
    }

    /// Blocks registered so far.
    pub fn blocks(&self) -> &BlockRegistry {
        &self.blocks
    }

    pub fn build(self) -> Content {
        Content {
            blocks: self.blocks,
            textures: self.textures,
            maps: self.maps,
        }
    }
}

impl Default for ContentBuilder {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ContentError {
    #[error("block `{0}` is registered twice")]
    DuplicateBlock(ResourceId),
    #[error("texture `{0}` is registered twice")]
    DuplicateTexture(ResourceId),
    #[error("block `{0}` is not registered")]
    MissingBlock(ResourceId),
    #[error("texture `{0}` is not registered")]
    MissingTexture(ResourceId),
    #[error(transparent)]
    InvalidId(#[from] InvalidId),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_duplicate_textures() {
        let mut builder = ContentBuilder::new();
        let id = ResourceId::new("base", "stone").unwrap();
        builder.add_texture(id.clone(), &b"png"[..]).unwrap();
        assert!(builder.add_texture(id.clone(), &b"png"[..]).is_err());
        assert_eq!(builder.build().texture(&id), Some(&b"png"[..]));
    }

    #[test]
    fn maps_belong_to_a_texture() {
        let mut builder = ContentBuilder::new();
        let id = ResourceId::new("base", "stone").unwrap();
        let maps = || TextureMaps {
            normal: (&b"normal"[..]).into(),
            specular: (&b"specular"[..]).into(),
        };
        assert!(builder.add_texture_maps(id.clone(), maps()).is_err());
        builder.add_texture(id.clone(), &b"png"[..]).unwrap();
        builder.add_texture_maps(id.clone(), maps()).unwrap();
        assert!(builder.add_texture_maps(id.clone(), maps()).is_err());
        let content = builder.build();
        assert_eq!(&*content.texture_maps(&id).unwrap().normal, b"normal");
    }
}
