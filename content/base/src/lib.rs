//! The base game: its blocks, their textures and the terrain they make up.
//!
//! It is built only on the public engine API, the same way a mod would be.

use ruda_core::{
    Appearance, BlockDef, BlockRegistry, ContentBuilder, ContentError, CubeTextures, ResourceId,
    TextureMaps, WorldBounds,
};
use ruda_worldgen::{Ore, TerrainGenerator, TerrainSettings};

pub const NAMESPACE: &str = "base";

/// Blocks a player can place, in hotbar order.
pub const HOTBAR: [&str; 11] = [
    "stone",
    "cobblestone",
    "dirt",
    "grass",
    "sand",
    "gravel",
    "planks",
    "copper_ore",
    "iron_ore",
    "lamp",
    "torch",
];

/// The sun and the moon, for the renderer's sky.
pub const SUN: &[u8] = include_bytes!("../sky/sun.png");
pub const MOON: &[u8] = include_bytes!("../sky/moon.png");

macro_rules! textures {
    ($($name:literal),* $(,)?) => {
        [$(($name, include_bytes!(concat!("../textures/", $name, ".png")) as &[u8])),*]
    };
}

const TEXTURES: [(&str, &[u8]); 13] = textures![
    "bedrock",
    "cobblestone",
    "copper_ore",
    "dirt",
    "grass_side",
    "grass_top",
    "gravel",
    "iron_ore",
    "lamp",
    "planks",
    "sand",
    "stone",
    "torch",
];

macro_rules! maps {
    ($($name:literal),* $(,)?) => {
        [$((
            $name,
            include_bytes!(concat!("../textures/", $name, "_n.png")) as &[u8],
            include_bytes!(concat!("../textures/", $name, "_s.png")) as &[u8],
        )),*]
    };
}

/// How textures take light: `<name>_n.png` and `<name>_s.png`, see
/// [`TextureMaps`]. The torch has none.
const MAPS: [(&str, &[u8], &[u8]); 12] = maps![
    "bedrock",
    "cobblestone",
    "copper_ore",
    "dirt",
    "grass_side",
    "grass_top",
    "gravel",
    "iron_ore",
    "lamp",
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
    for (name, normal, specular) in MAPS {
        let maps = TextureMaps {
            normal: normal.into(),
            specular: specular.into(),
        };
        content.add_texture_maps(id(name)?, maps)?;
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
    let bedrock = CubeTextures::all(id("bedrock")?);
    content.add_block(BlockDef::new(id("bedrock")?, Appearance::Cube(bedrock)).unbreakable())?;
    // A warm, amber light.
    content.add_block(
        BlockDef::new(
            id("torch")?,
            Appearance::Torch {
                texture: id("torch")?,
            },
        )
        .emits_light(14, 11, 6),
    )?;
    let lamp = CubeTextures::all(id("lamp")?);
    content.add_block(BlockDef::new(id("lamp")?, Appearance::Cube(lamp)).emits_light(15, 12, 7))?;
    Ok(())
}

/// The base game's terrain for `seed`, built from the blocks of [`register`].
/// Sea level is at 0 and an unbreakable floor of bedrock lies at the bottom
/// of `bounds`.
pub fn terrain(
    blocks: &BlockRegistry,
    seed: u64,
    bounds: WorldBounds,
) -> Result<TerrainGenerator, ContentError> {
    let block = |name| {
        let id = id(name)?;
        blocks.id(&id).ok_or(ContentError::MissingBlock(id))
    };
    let settings = TerrainSettings {
        sea_level: 0,
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
        floor: block("bedrock")?,
        min_y: bounds.min_y,
        max_y: bounds.max_y,
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
            let textures: Vec<&ResourceId> = match &def.appearance {
                Appearance::Cube(textures) => Face::ALL
                    .iter()
                    .map(|&face| textures.for_face(face))
                    .collect(),
                Appearance::Torch { texture } => vec![texture],
                Appearance::Invisible => continue,
            };
            for texture in textures {
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
    fn textures_are_square_pngs_with_maps_to_match() {
        let size = |png: &[u8]| {
            assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
            let width = u32::from_be_bytes(png[16..20].try_into().unwrap());
            let height = u32::from_be_bytes(png[20..24].try_into().unwrap());
            (width, height)
        };
        for (name, png) in TEXTURES {
            let (width, height) = size(png);
            assert!(width == height && [16, 32, 64].contains(&width), "{name}");
            if let Some((_, normal, specular)) = MAPS.iter().find(|(map, ..)| *map == name) {
                assert_eq!(size(normal), (width, height), "{name}");
                assert_eq!(size(specular), (width, height), "{name}");
            }
        }
    }

    #[test]
    fn hotbar_blocks_exist_and_terrain_builds() {
        let content = content();
        for name in HOTBAR {
            assert!(content.blocks().id(&id(name).unwrap()).is_some(), "{name}");
        }
        terrain(content.blocks(), 1, WorldBounds::DEFAULT).unwrap();
        let bedrock = content.blocks().id(&id("bedrock").unwrap()).unwrap();
        assert!(!content.blocks().is_breakable(bedrock));
        let lamp = content.blocks().id(&id("lamp").unwrap()).unwrap();
        assert!(!content.blocks().light(lamp).is_dark());
        for png in [SUN, MOON] {
            assert!(png.starts_with(b"\x89PNG\r\n\x1a\n"));
        }
    }
}
