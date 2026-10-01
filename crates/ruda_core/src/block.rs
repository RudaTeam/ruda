use std::collections::HashMap;

use crate::{ContentError, Face, ResourceId};

/// Numeric id of a registered block. Ids are only valid within one session:
/// the server assigns them when content is registered and tells clients the
/// mapping, while saves and the content API use [`ResourceId`]s.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, PartialOrd, Ord)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct BlockId(u32);

impl BlockId {
    pub const AIR: Self = Self(0);
    /// Stands in for blocks whose content is missing, e.g. from a removed mod.
    pub const UNKNOWN: Self = Self(1);

    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    pub const fn raw(self) -> u32 {
        self.0
    }

    pub fn index(self) -> usize {
        self.0 as usize
    }
}

/// How a block looks.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Appearance {
    /// Not drawn and not solid, like air.
    Invisible,
    /// A full opaque cube.
    Cube(CubeTextures),
}

/// Textures of a cube block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CubeTextures {
    pub top: ResourceId,
    pub bottom: ResourceId,
    pub side: ResourceId,
}

impl CubeTextures {
    /// The same texture on every face.
    pub fn all(texture: ResourceId) -> Self {
        Self {
            top: texture.clone(),
            bottom: texture.clone(),
            side: texture,
        }
    }

    pub fn for_face(&self, face: Face) -> &ResourceId {
        match face {
            Face::PosY => &self.top,
            Face::NegY => &self.bottom,
            _ => &self.side,
        }
    }
}

/// A registered kind of block.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BlockDef {
    pub id: ResourceId,
    pub appearance: Appearance,
    /// Whether players can break it. Unbreakable blocks can't be placed by
    /// players either.
    pub breakable: bool,
}

impl BlockDef {
    pub fn new(id: ResourceId, appearance: Appearance) -> Self {
        Self {
            id,
            appearance,
            breakable: true,
        }
    }

    pub fn unbreakable(mut self) -> Self {
        self.breakable = false;
        self
    }

    /// Whether the block fills its whole cell: it hides the faces of its
    /// neighbours and stops movement.
    pub fn is_solid(&self) -> bool {
        matches!(self.appearance, Appearance::Cube(_))
    }
}

/// Every registered block, indexed by [`BlockId`].
#[derive(Clone, Debug)]
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    ids: HashMap<ResourceId, BlockId>,
}

impl BlockRegistry {
    /// A registry holding only the engine's own blocks.
    pub(crate) fn with_engine_blocks() -> Self {
        let engine_id = |path| ResourceId::new(ResourceId::ENGINE, path).expect("valid id");
        let mut registry = Self {
            defs: Vec::new(),
            ids: HashMap::new(),
        };
        let air = registry.register(BlockDef::new(engine_id("air"), Appearance::Invisible));
        let unknown = registry.register(BlockDef::new(
            engine_id("unknown"),
            Appearance::Cube(CubeTextures::all(engine_id("unknown"))),
        ));
        debug_assert_eq!(
            (air.ok(), unknown.ok()),
            (Some(BlockId::AIR), Some(BlockId::UNKNOWN))
        );
        registry
    }

    pub(crate) fn register(&mut self, def: BlockDef) -> Result<BlockId, ContentError> {
        if self.ids.contains_key(&def.id) {
            return Err(ContentError::DuplicateBlock(def.id));
        }
        let id = BlockId(u32::try_from(self.defs.len()).expect("fewer than 2^32 blocks"));
        self.ids.insert(def.id.clone(), id);
        self.defs.push(def);
        Ok(id)
    }

    /// The block's definition. Ids this registry does not know resolve to
    /// [`BlockId::UNKNOWN`].
    pub fn get(&self, id: BlockId) -> &BlockDef {
        self.defs
            .get(id.index())
            .unwrap_or(&self.defs[BlockId::UNKNOWN.index()])
    }

    pub fn id(&self, name: &ResourceId) -> Option<BlockId> {
        self.ids.get(name).copied()
    }

    pub fn is_solid(&self, id: BlockId) -> bool {
        self.get(id).is_solid()
    }

    pub fn is_breakable(&self, id: BlockId) -> bool {
        self.get(id).breakable
    }

    pub fn len(&self) -> usize {
        self.defs.len()
    }

    /// Always false: the engine's own blocks are always registered.
    pub fn is_empty(&self) -> bool {
        self.defs.is_empty()
    }

    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &BlockDef)> {
        self.defs
            .iter()
            .enumerate()
            .map(|(index, def)| (BlockId(index as u32), def))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn engine_blocks_come_first() {
        let registry = BlockRegistry::with_engine_blocks();
        assert_eq!(registry.get(BlockId::AIR).id.as_str(), "ruda:air");
        assert!(!registry.is_solid(BlockId::AIR));
        assert_eq!(registry.get(BlockId::UNKNOWN).id.as_str(), "ruda:unknown");
        assert!(registry.is_solid(BlockId::UNKNOWN));
    }

    #[test]
    fn unknown_ids_resolve_to_the_placeholder() {
        let registry = BlockRegistry::with_engine_blocks();
        assert_eq!(
            registry.get(BlockId::from_raw(999)).id.as_str(),
            "ruda:unknown"
        );
    }

    #[test]
    fn rejects_duplicates() {
        let mut registry = BlockRegistry::with_engine_blocks();
        let stone = ResourceId::new("base", "stone").unwrap();
        let def = BlockDef::new(stone.clone(), Appearance::Cube(CubeTextures::all(stone)));
        let id = registry.register(def.clone()).unwrap();
        assert_eq!(registry.id(&def.id), Some(id));
        assert!(matches!(
            registry.register(def),
            Err(ContentError::DuplicateBlock(_))
        ));
    }
}
