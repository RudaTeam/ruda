//! A closed room of stone lit by a red lamp on the left and a blue one on
//! the right, drawn off-screen: each side takes its lamp's colour.
//!
//! Needs a GPU; without one the test only says so and passes. With
//! `RUDA_TEST_IMAGES` set to a directory, the picture is saved there.

use glam::DVec3;
use ruda_core::{
    Appearance, BlockDef, BlockPos, ChunkPos, ContentBuilder, CubeTextures, Light, ResourceId,
    WorldBounds,
};
use ruda_render::{Backdrop, Camera, PaddedChunk, Renderer, Scene, mesh_chunk};
use ruda_world::light::LightEngine;
use ruda_world::{Chunk, World};

const WIDTH: u32 = 480;
const HEIGHT: u32 = 270;

#[test]
fn coloured_lamps_light_their_side_of_a_room() {
    let mut renderer = match pollster::block_on(Renderer::headless(WIDTH, HEIGHT)) {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skipping, no GPU: {error:#}");
            return;
        }
    };
    let mut content = ContentBuilder::new();
    ruda_base::register(&mut content).unwrap();
    let lamp_texture = ruda_base::id("lamp").unwrap();
    let mut lamp = |name: &str, light: Light| {
        let id: ResourceId = format!("test:{name}").parse().unwrap();
        let def = BlockDef::new(
            id,
            Appearance::Cube(CubeTextures::all(lamp_texture.clone())),
        );
        content
            .add_block(def.emits_light(light.red(), light.green(), light.blue()))
            .unwrap()
    };
    let red = lamp("red", Light::rgb(15, 1, 1));
    let blue = lamp("blue", Light::rgb(1, 2, 15));
    let content = content.build();
    let stone = content
        .blocks()
        .id(&ruda_base::id("stone").unwrap())
        .unwrap();

    // Stone all around a room inside the middle chunk.
    let mut world = World::new();
    for y in -1..=1 {
        for z in -1..=1 {
            for x in -1..=1 {
                world.insert_chunk(ChunkPos::new(x, y, z), Chunk::filled(stone));
            }
        }
    }
    for y in 2..14 {
        for z in 2..30 {
            for x in 2..30 {
                world.set_block(BlockPos::new(x, y, z), ruda_core::BlockId::AIR);
            }
        }
    }
    // A pillar between the lamps.
    for y in 2..10 {
        world.set_block(BlockPos::new(16, y, 18), stone);
    }
    // Looking towards +z, +x is on the left.
    world.set_block(BlockPos::new(27, 3, 18), red);
    world.set_block(BlockPos::new(4, 3, 18), blue);

    let bounds = WorldBounds {
        min_y: -32,
        max_y: 63,
    };
    let mut engine = LightEngine::new(content.blocks(), bounds);
    for y in (-1..=1).rev() {
        for z in -1..=1 {
            for x in -1..=1 {
                engine.light_chunk(&mut world, ChunkPos::new(x, y, z));
            }
        }
    }

    let faces = renderer.load_block_textures(&content);
    let positions: Vec<ChunkPos> = world.chunks().map(|(pos, _)| pos).collect();
    for pos in positions {
        let chunk = PaddedChunk::gather(&world, pos).unwrap();
        renderer.upload_chunk(pos, &mesh_chunk(&chunk, &faces));
    }

    // From the near wall, looking across the room at the pillar.
    let mut camera = Camera::new(DVec3::new(16.0, 8.0, 3.0));
    camera.rotate(std::f32::consts::PI, -0.35);
    let scene = Scene {
        camera,
        target: None,
        view_distance: 64.0,
        bounds: Some(bounds),
        time_of_day: 0.75,
        eye_light: Light::DARK,
        lod_distance: 0.0,
    };
    let (width, height, pixels) = renderer.capture(Backdrop::World(&scene), None).unwrap();
    if let Some(dir) = std::env::var_os("RUDA_TEST_IMAGES") {
        save(
            &std::path::Path::new(&dir).join("lighting.png"),
            width,
            height,
            &pixels,
        );
    }

    // Average colour of a band of columns.
    let band = |from: u32, to: u32| {
        let mut sum = [0u64; 3];
        for y in 0..height {
            for x in from..to {
                let at = ((y * width + x) * 4) as usize;
                for (sum, &value) in sum.iter_mut().zip(&pixels[at..at + 3]) {
                    *sum += u64::from(value);
                }
            }
        }
        sum
    };
    let left = band(0, width / 4);
    let right = band(width * 3 / 4, width);
    assert!(left[0] > left[2] * 2, "the left side is red: {left:?}");
    assert!(right[2] > right[0] * 2, "the right side is blue: {right:?}");
}

fn save(path: &std::path::Path, width: u32, height: u32, rgba: &[u8]) {
    let file = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder
        .write_header()
        .unwrap()
        .write_image_data(rgba)
        .unwrap();
}
