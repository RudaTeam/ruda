use std::collections::HashMap;

use crate::{ContentError, Face, Light, ResourceId};

/// Numeric id of a registered block in one of its states. Ids are only valid
/// within one session: the server assigns them when content is registered
/// and tells clients the mapping, while saves and the content API use
/// [`ResourceId`]s.
///
/// A block with several states, like a torch that can stand on the floor or
/// hang on a wall, takes one id per state, in a row; its first is the id of
/// the block itself.
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
    /// A torch: a thin stick that stands on the floor or leans out of a
    /// wall, see [`Mount`]. The texture holds the stick in its middle two
    /// columns, flame on top.
    Torch { texture: ResourceId },
}

/// Where a torch is fixed: its state.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Mount {
    /// Standing on the block below.
    Floor,
    /// On the side of a wall, facing away from it: `Wall(Face::PosX)` hangs
    /// on the block at −x.
    Wall(Face),
}

impl Mount {
    /// Floor, then walls facing +x, −x, +z, −z.
    pub const ALL: [Mount; 5] = [
        Mount::Floor,
        Mount::Wall(Face::PosX),
        Mount::Wall(Face::NegX),
        Mount::Wall(Face::PosZ),
        Mount::Wall(Face::NegZ),
    ];

    /// Placed against the `face` of a block: on top of it or on its side.
    /// Torches don't hang from ceilings.
    pub fn against(face: Face) -> Option<Self> {
        match face {
            Face::PosY => Some(Mount::Floor),
            Face::NegY => None,
            side => Some(Mount::Wall(side)),
        }
    }

    /// The direction of the block holding it up.
    pub fn support(self) -> Face {
        match self {
            Mount::Floor => Face::NegY,
            Mount::Wall(facing) => facing.opposite(),
        }
    }

    fn state(self) -> u8 {
        Mount::ALL
            .iter()
            .position(|&mount| mount == self)
            .expect("every mount is listed") as u8
    }
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
    /// The light it gives off; its sky channel is unused.
    pub light: Light,
}

impl BlockDef {
    pub fn new(id: ResourceId, appearance: Appearance) -> Self {
        Self {
            id,
            appearance,
            breakable: true,
            light: Light::DARK,
        }
    }

    pub fn unbreakable(mut self) -> Self {
        self.breakable = false;
        self
    }

    /// Makes the block give off light of this colour, each channel 0 to 15.
    pub fn emits_light(mut self, red: u8, green: u8, blue: u8) -> Self {
        self.light = Light::rgb(red, green, blue);
        self
    }

    /// Whether the block fills its whole cell: it hides the faces of its
    /// neighbours and stops movement.
    pub fn is_solid(&self) -> bool {
        matches!(self.appearance, Appearance::Cube(_))
    }

    /// Whether it is drawn at all: everything but air.
    pub fn is_visible(&self) -> bool {
        !matches!(self.appearance, Appearance::Invisible)
    }

    /// How many states the block has, each with an id of its own.
    pub fn states(&self) -> u8 {
        match self.appearance {
            Appearance::Torch { .. } => Mount::ALL.len() as u8,
            _ => 1,
        }
    }

    /// Whether light can't pass through it.
    pub fn is_opaque(&self) -> bool {
        self.is_solid()
    }
}

/// Every registered block, indexed by [`BlockId`].
#[derive(Clone, Debug)]
pub struct BlockRegistry {
    defs: Vec<BlockDef>,
    /// For every id, its block's index in `defs` and its state.
    states: Vec<(u32, u8)>,
    ids: HashMap<ResourceId, BlockId>,
}

impl BlockRegistry {
    /// A registry holding only the engine's own blocks.
    pub(crate) fn with_engine_blocks() -> Self {
        let engine_id = |path| ResourceId::new(ResourceId::ENGINE, path).expect("valid id");
        let mut registry = Self {
            defs: Vec::new(),
            states: Vec::new(),
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
        let id = BlockId(u32::try_from(self.states.len()).expect("fewer than 2^32 blocks"));
        let index = u32::try_from(self.defs.len()).expect("fewer than 2^32 blocks");
        for state in 0..def.states() {
            self.states.push((index, state));
        }
        self.ids.insert(def.id.clone(), id);
        self.defs.push(def);
        Ok(id)
    }

    /// The block's definition. Ids this registry does not know resolve to
    /// [`BlockId::UNKNOWN`].
    pub fn get(&self, id: BlockId) -> &BlockDef {
        let index = self.states.get(id.index()).map_or(1, |&(index, _)| index);
        &self.defs[index as usize]
    }

    /// Which of its block's states the id is.
    pub fn state(&self, id: BlockId) -> u8 {
        self.states.get(id.index()).map_or(0, |&(_, state)| state)
    }

    /// The id of the block in its first state.
    pub fn base(&self, id: BlockId) -> BlockId {
        BlockId(id.0 - u32::from(self.state(id)))
    }

    /// Where a torch is fixed, for torch ids.
    pub fn mount(&self, id: BlockId) -> Option<Mount> {
        match self.get(id).appearance {
            Appearance::Torch { .. } => Mount::ALL.get(usize::from(self.state(id))).copied(),
            _ => None,
        }
    }

    /// The id to place for `block` put against the `face` of another block:
    /// a torch on the floor or a wall. `None` if it can't go there.
    pub fn placed(&self, block: BlockId, face: Face) -> Option<BlockId> {
        match self.get(block).appearance {
            Appearance::Torch { .. } => {
                let mount = Mount::against(face)?;
                Some(BlockId(self.base(block).0 + u32::from(mount.state())))
            }
            _ => Some(block),
        }
    }

    /// The direction of the block that has to be there to hold this one up.
    pub fn support(&self, id: BlockId) -> Option<Face> {
        self.mount(id).map(Mount::support)
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

    pub fn is_visible(&self, id: BlockId) -> bool {
        self.get(id).is_visible()
    }

    pub fn is_opaque(&self, id: BlockId) -> bool {
        self.get(id).is_opaque()
    }

    /// The light the block gives off.
    pub fn light(&self, id: BlockId) -> Light {
        self.get(id).light
    }

    /// Number of ids, a state counting as one.
    pub fn len(&self) -> usize {
        self.states.len()
    }

    /// Always false: the engine's own blocks are always registered.
    pub fn is_empty(&self) -> bool {
        self.states.is_empty()
    }

    /// Every id with its block, states included.
    pub fn iter(&self) -> impl Iterator<Item = (BlockId, &BlockDef)> {
        self.states
            .iter()
            .enumerate()
            .map(|(id, &(index, _))| (BlockId(id as u32), &self.defs[index as usize]))
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
    fn torches_take_an_id_per_mount() {
        let mut registry = BlockRegistry::with_engine_blocks();
        let torch_id = ResourceId::new("base", "torch").unwrap();
        let torch = registry
            .register(BlockDef::new(
                torch_id.clone(),
                Appearance::Torch { texture: torch_id },
            ))
            .unwrap();
        let stone_id = ResourceId::new("base", "stone").unwrap();
        let stone = registry
            .register(BlockDef::new(
                stone_id.clone(),
                Appearance::Cube(CubeTextures::all(stone_id)),
            ))
            .unwrap();
        assert_eq!(stone.raw(), torch.raw() + 5);
        assert_eq!(registry.len(), 2 + 5 + 1);

        let on_wall = registry.placed(torch, Face::NegZ).unwrap();
        assert_eq!(registry.base(on_wall), torch);
        assert_eq!(registry.mount(on_wall), Some(Mount::Wall(Face::NegZ)));
        assert_eq!(registry.support(on_wall), Some(Face::PosZ));
        assert_eq!(registry.support(torch), Some(Face::NegY));
        assert_eq!(registry.placed(torch, Face::NegY), None);
        assert_eq!(registry.placed(stone, Face::NegY), Some(stone));
        assert_eq!(registry.get(on_wall).id.as_str(), "base:torch");
        assert!(!registry.is_solid(on_wall) && registry.is_visible(on_wall));
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
