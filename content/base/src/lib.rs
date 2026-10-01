//! The base game: its blocks, their textures and the terrain they make up.
//!
//! It is built only on the public engine API, the same way a mod would be.

use ruda_core::{
    Appearance, BlockDef, BlockRegistry, ContentBuilder, ContentError, CubeTextures, ResourceId,
};
use ruda_worldgen::{Ore, TerrainGenerator, TerrainSettings};

pub const NAMESPACE: &str = "base";

/// Blocks a player can place, in hotbar order.
pub const HOTBAR: [&str; 9] = [
    "stone",
    "cobblestone",
    "dirt",
    "grass",
    "sand",
    "gravel",
    "planks",
    "copper_ore",
    "iron_ore",
];

macro_rules! textures {
    ($($name:literal),* $(,)?) => {
        [$(($name, include_bytes!(concat!("../textures/", $name, ".png")) as &[u8])),*]
    };
}

const TEXTURES: [(&str, &[u8]); 10] = textures![
    "cobblestone",
    "copper_ore",
    "dirt",
    "grass_side",
    "grass_top",
    "gravel",
    "iron_ore",
    "planks",
    "sand",
    "stone",
];

/// Blocks that look the same from every side, with a texture of the same name.
const PLAIN_BLOCKS: [&str; 8] = [
    "stone",
    "cobblestone",
    "dirt",
    "sand",
    "gravel",
    "planks",
    "copper_ore",
    "iron_ore",
];

/// `base:<path>`.
pub fn id(path: &str) -> Result<ResourceId, ContentError> {
    Ok(ResourceId::new(NAMESPACE, path)?)
}

/// Registers the base game's textures and blocks.
pub fn register(content: &mut ContentBuilder) -> Result<(), ContentError> {
    for (name, png) in TEXTURES {
        content.add_texture(id(name)?, png)?;
    }
    for name in PLAIN_BLOCKS {
        let textures = CubeTextures::all(id(name)?);
        content.add_block(BlockDef::new(id(name)?, Appearance::Cube(textures)))?;
    }
    let grass = CubeTextures {
        top: id("grass_top")?,
        bottom: id("dirt")?,
        side: id("grass_side")?,
    };
    content.add_block(BlockDef::new(id("grass")?, Appearance::Cube(grass)))?;
    Ok(())
}

/// The base game's terrain for `seed`, built from the blocks of [`register`].
pub fn terrain(blocks: &BlockRegistry, seed: u64) -> Result<TerrainGenerator, ContentError> {
    let block = |name| {
        let id = id(name)?;
        blocks.id(&id).ok_or(ContentError::MissingBlock(id))
    };
    let settings = TerrainSettings {
        sea_level: 32,
        grass: block("grass")?,
        dirt: block("dirt")?,
        sand: block("sand")?,
        stone: block("stone")?,
        ores: vec![
            Ore {
                block: block("copper_ore")?,
                min_y: -256,
                max_y: 96,
                threshold: 0.65,
            },
            Ore {
                block: block("iron_ore")?,
                min_y: -512,
                max_y: 32,
                threshold: 0.78,
            },
        ],
    };
    Ok(TerrainGenerator::new(seed, settings))
}

#[cfg(test)]
mod tests {
    use ruda_core::{Content, Face};

    use super::*;

    fn content() -> Content {
        let mut builder = ContentBuilder::new();
        register(&mut builder).unwrap();
        builder.build()
    }

    #[test]
    fn every_block_texture_is_registered() {
        let content = content();
        for (_, def) in content.blocks().iter() {
            let Appearance::Cube(textures) = &def.appearance else {
                continue;
            };
            for face in Face::ALL {
                let texture = textures.for_face(face);
                if texture.namespace() == NAMESPACE {
                    assert!(
                        content.texture(texture).is_some(),
                        "{} needs {texture}",
                        def.id
                    );
                }
            }
        }
    }

    #[test]
    fn textures_are_16_pixel_pngs() {
        for (name, png) in TEXTURES {
            assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"), "{name}");
            let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
            let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
            assert_eq!((width, height), (16, 16), "{name}");
        }
    }

    #[test]
    fn hotbar_blocks_exist_and_terrain_builds() {
        let content = content();
        for name in HOTBAR {
            assert!(content.blocks().id(&id(name).unwrap()).is_some(), "{name}");
        }
        terrain(content.blocks(), 1).unwrap();
    }
}
