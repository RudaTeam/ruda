//! Shared vocabulary of the engine: content identifiers, block coordinates
//! and the registries that content packs fill in at startup.
//!
//! The engine itself knows no game content besides air and a placeholder for
//! unknown blocks; everything else is registered by content packs.

mod block;
mod bounds;
mod content;
mod face;
mod id;
mod light;
mod pos;

pub use block::{Appearance, BlockDef, BlockId, BlockRegistry, CubeTextures};
pub use bounds::WorldBounds;
pub use content::{Content, ContentBuilder, ContentError};
pub use face::Face;
pub use id::{InvalidId, ResourceId};
pub use light::Light;
pub use pos::{BlockPos, CHUNK_SHIFT, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, LocalPos};
