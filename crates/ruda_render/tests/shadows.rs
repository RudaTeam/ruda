//! A stone pillar on flat ground under the morning sun: with shadows on, the
//! ground in its shadow is darker than the ground in the sun; with shadows
//! off, both look the same.
//!
//! Needs a GPU; without one the test only says so and passes. With
//! `RUDA_TEST_IMAGES` set to a directory, the pictures are saved there.

use glam::{DVec3, Vec3};
use ruda_core::{BlockPos, ChunkPos, ContentBuilder, Light, WorldBounds};
use ruda_render::{Backdrop, Camera, PaddedChunk, Renderer, Scene, mesh_chunk};
use ruda_world::light::LightEngine;
use ruda_world::{Chunk, World};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
/// Mid-morning: the sun is in the east, a third of the way up.
const TIME_OF_DAY: f32 = 0.1;

#[test]
fn a_pillar_casts_a_shadow() {
    let mut renderer = match pollster::block_on(Renderer::headless(WIDTH, HEIGHT)) {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skipping, no GPU: {error:#}");
            return;
        }
    };
    let mut content = ContentBuilder::new();
    ruda_base::register(&mut content).unwrap();
    let content = content.build();
    let block = |name| content.blocks().id(&ruda_base::id(name).unwrap()).unwrap();
    let (stone, sand) = (block("stone"), block("sand"));

    // Sand ground with its top at y = 0, and a 3×3 pillar of stone, 12 tall.
    let bounds = WorldBounds {
        min_y: -32,
        max_y: 63,
    };
    let mut world = World::new();
    for y in -1..=1 {
        for z in -2..=1 {
            for x in -2..=1 {
                let fill = if y < 0 { sand } else { ruda_core::BlockId::AIR };
                world.insert_chunk(ChunkPos::new(x, y, z), Chunk::filled(fill));
            }
        }
    }
    for y in 0..12 {
        for z in 15..18 {
            for x in 15..18 {
                world.set_block(BlockPos::new(x, y, z), stone);
            }
        }
    }
    let mut engine = LightEngine::new(content.blocks(), bounds);
    for y in (-1..=1).rev() {
        for z in -2..=1 {
            for x in -2..=1 {
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

    // The sun's direction, as the renderer works it out.
    let angle = TIME_OF_DAY * std::f32::consts::TAU;
    let sun = Vec3::new(angle.cos(), angle.sin(), 0.25).normalize();
    // The shadow of the pillar's top lands this far from its foot. Halfway
    // there is in shadow; as far the other way is in the sun.
    let reach = -sun * (12.0 / sun.y);
    let foot = Vec3::new(16.5, 0.0, 16.5);
    let shaded = foot + Vec3::new(reach.x, 0.0, reach.z) * 0.5;
    let sunny = foot - Vec3::new(reach.x, 0.0, reach.z) * 0.5;

    // From above and south of the pillar, looking down at it.
    let mut camera = Camera::new(DVec3::new(12.0, 28.0, 36.0));
    camera.rotate(0.0, -0.95);
    let scene = Scene {
        camera,
        target: None,
        view_distance: 256.0,
        bounds: Some(bounds),
        time_of_day: TIME_OF_DAY,
        eye_light: Light::SKY,
    };

    let mut brightness = |shadows: bool| {
        renderer.set_shadows(shadows);
        let (width, height, pixels) = renderer.capture(Backdrop::World(&scene), None).unwrap();
        if let Some(dir) = std::env::var_os("RUDA_TEST_IMAGES") {
            let name = if shadows {
                "shadows-on.png"
            } else {
                "shadows-off.png"
            };
            save(
                &std::path::Path::new(&dir).join(name),
                width,
                height,
                &pixels,
            );
        }
        let view_proj = camera.view_proj(width as f32 / height as f32, 320.0);
        [shaded, sunny].map(|point| {
            let clip = view_proj.project_point3((point.as_dvec3() - camera.position).as_vec3());
            let x = ((clip.x * 0.5 + 0.5) * width as f32) as u32;
            let y = ((0.5 - clip.y * 0.5) * height as f32) as u32;
            assert!(x < width && y < height, "{point} is off screen");
            // Average over a few pixels around the point.
            let mut sum = 0u32;
            for dy in 0..3 {
                for dx in 0..3 {
                    let at = (((y + dy - 1) * width + x + dx - 1) * 4) as usize;
                    sum += pixels[at..at + 3]
                        .iter()
                        .map(|&c| u32::from(c))
                        .sum::<u32>();
                }
            }
            sum as f32 / 27.0
        })
    };
    let [shaded_off, sunny_off] = brightness(false);
    let [shaded_on, sunny_on] = brightness(true);
    assert!(
        (shaded_off - sunny_off).abs() < sunny_off * 0.1,
        "without shadows the ground looks the same: {shaded_off} and {sunny_off}"
    );
    assert!(
        shaded_on < sunny_on * 0.8,
        "with shadows the ground behind the pillar is darker: {shaded_on} and {sunny_on}"
    );
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
