use std::borrow::Cow;
use std::collections::BTreeMap;

use crate::{BlockDef, BlockId, BlockRegistry, InvalidId, ResourceId};

/// Everything content packs registered for a session: blocks and the
/// textures they use. Built once with [`ContentBuilder`], immutable afterwards.
#[derive(Debug)]
pub struct Content {
    blocks: BlockRegistry,
    textures: BTreeMap<ResourceId, Cow<'static, [u8]>>,
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
}

/// Collects content from the engine and content packs at startup.
#[derive(Debug)]
pub struct ContentBuilder {
    blocks: BlockRegistry,
    textures: BTreeMap<ResourceId, Cow<'static, [u8]>>,
}

impl ContentBuilder {
    pub fn new() -> Self {
        Self {
            blocks: BlockRegistry::with_engine_blocks(),
            textures: BTreeMap::new(),
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

    /// Blocks registered so far.
    pub fn blocks(&self) -> &BlockRegistry {
        &self.blocks
    }

    pub fn build(self) -> Content {
        Content {
            blocks: self.blocks,
            textures: self.textures,
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
}
