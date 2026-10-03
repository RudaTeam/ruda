//! Clouds over flat grass: they shade the ground, and they part around a
//! tower that reaches up through them.
//!
//! Needs a GPU; without one the test only says so and passes. With
//! `RUDA_TEST_IMAGES` set to a directory, the pictures are saved there.

use glam::{DVec2, DVec3, Vec3};
use ruda_core::{BlockId, BlockPos, ChunkPos, ContentBuilder, Light, WorldBounds};
use ruda_render::{
    Backdrop, Camera, CloudSky, PaddedChunk, Renderer, Scene, cloud_obstacles, mesh_chunk,
};
use ruda_world::light::LightEngine;
use ruda_world::{Chunk, World};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
/// Overcast: thick cloud almost everywhere.
const SKY: CloudSky = CloudSky {
    seed: 1,
    cover: 1.0,
    density: 1.0,
    drift: DVec2::ZERO,
};

#[test]
fn clouds_shade_the_ground_and_part_around_a_tower() {
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
    let (stone, grass) = (block("stone"), block("grass"));

    // Grass with its top at y = 0, and a 3×3 tower from there to the top of
    // the world, through the clouds at 221 to 227.
    let bounds = WorldBounds {
        min_y: -32,
        max_y: 255,
    };
    let mut world = World::new();
    for y in -1..=7 {
        for z in -2..=2 {
            for x in -2..=2 {
                let fill = if y < 0 { grass } else { BlockId::AIR };
                world.insert_chunk(ChunkPos::new(x, y, z), Chunk::filled(fill));
            }
        }
    }
    for y in 0..=255 {
        for z in 15..18 {
            for x in 15..18 {
                world.set_block(BlockPos::new(x, y, z), stone);
            }
        }
    }
    let mut engine = LightEngine::new(content.blocks(), bounds);
    for y in (-1..=7).rev() {
        for z in -2..=2 {
            for x in -2..=2 {
                engine.light_chunk(&mut world, ChunkPos::new(x, y, z));
            }
        }
    }
    let faces = renderer.load_block_textures(&content);
    let positions: Vec<ChunkPos> = world.chunks().map(|(pos, _)| pos).collect();
    for &pos in &positions {
        let chunk = PaddedChunk::gather(&world, pos).unwrap();
        renderer.upload_chunk(pos, &mesh_chunk(&chunk, &faces));
    }

    let scene = |camera: Camera, clouds: Option<CloudSky>| Scene {
        camera,
        target: None,
        crosshair: true,
        view_distance: 512.0,
        bounds: Some(bounds),
        time_of_day: 0.25,
        eye_light: Light::SKY,
        lod_distance: 0.0,
        clouds,
    };
    let pixel = |renderer: &mut Renderer, scene: &Scene, point: Vec3, name: &str| {
        let (width, height, pixels) = renderer.capture(Backdrop::World(scene), None).unwrap();
        if let Some(dir) = std::env::var_os("RUDA_TEST_IMAGES") {
            save(
                &std::path::Path::new(&dir).join(name),
                width,
                height,
                &pixels,
            );
        }
        let camera = scene.camera;
        let view_proj = camera.view_proj(width as f32 / height as f32, 320.0);
        let clip = view_proj.project_point3((point.as_dvec3() - camera.position).as_vec3());
        let x = ((clip.x * 0.5 + 0.5) * width as f32) as u32;
        let y = ((0.5 - clip.y * 0.5) * height as f32) as u32;
        assert!(x < width && y < height, "{point} is off screen");
        let at = ((y * width + x) * 4) as usize;
        [pixels[at], pixels[at + 1], pixels[at + 2]].map(f32::from)
    };
    let brightness = |rgb: [f32; 3]| rgb.iter().sum::<f32>() / 3.0;
    let difference = |a: [f32; 3], b: [f32; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).sum::<f32>();

    let name = |what: &str| format!("clouds-{what}.png");

    // Under the clouds, looking down at the grass away from the tower.
    let mut below = Camera::new(DVec3::new(-30.0, 30.0, -30.0));
    below.rotate(0.0, -1.5);
    let ground = Vec3::new(-30.0, 0.0, -35.0);
    let clear = pixel(&mut renderer, &scene(below, None), ground, &name("none"));
    let shaded = pixel(
        &mut renderer,
        &scene(below, Some(SKY)),
        ground,
        &name("shade"),
    );
    assert!(
        brightness(shaded) < brightness(clear) * 0.85,
        "clouds darken the ground: {shaded:?} against {clear:?}"
    );

    // Above the clouds, looking down beside the tower, and far from it.
    let looking_down = |x, z| {
        let mut camera = Camera::new(DVec3::new(x, 262.0, z));
        camera.rotate(0.0, -1.5);
        scene(camera, Some(SKY))
    };
    let (above, above_far) = (looking_down(12.0, 12.0), looking_down(64.0, 12.0));
    let beside = Vec3::new(10.0, 0.0, 10.0);
    let far = Vec3::new(64.0, 0.0, 10.0);
    let unparted = pixel(&mut renderer, &above, beside, &name("unparted"));
    let unparted_far = pixel(&mut renderer, &above_far, far, &name("unparted-far"));
    for &pos in &positions {
        let chunk = world.chunk(pos).unwrap();
        let tops = cloud_obstacles(pos, chunk, |block| content.blocks().is_solid(block));
        renderer.set_cloud_obstacles(pos, tops);
    }
    // Changed obstacles are taken into account after a moment.
    std::thread::sleep(std::time::Duration::from_millis(600));
    let parted = pixel(&mut renderer, &above, beside, &name("parted"));
    let parted_far = pixel(&mut renderer, &above_far, far, &name("parted-far"));
    assert!(
        difference(parted, unparted) > 60.0,
        "clouds part beside the tower: {parted:?} against {unparted:?}"
    );
    assert!(
        difference(parted_far, unparted_far) < 30.0,
        "and stay away from it: {parted_far:?} against {unparted_far:?}"
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
