//! Clouds over flat sand: they shade the ground, and they part around a
//! pillar that reaches up into them.
//!
//! Needs a GPU; without one the test only says so and passes. With
//! `RUDA_TEST_IMAGES` set to a directory, the pictures are saved there.

use glam::{DVec3, Vec3};
use ruda_core::{BlockId, BlockPos, ChunkPos, ContentBuilder, Light, WorldBounds};
use ruda_render::{
    Backdrop, CLOUD_CELL, Camera, CloudSky, PaddedChunk, Renderer, Scene, cloud_at,
    cloud_obstacles, mesh_chunk,
};
use ruda_world::light::LightEngine;
use ruda_world::{Chunk, World};

const WIDTH: u32 = 640;
const HEIGHT: u32 = 360;
/// Overcast: every cell near the pillar holds a cloud.
const SKY: CloudSky = CloudSky {
    seed: 1,
    cover: 1.0,
    time: 0.0,
};

#[test]
fn clouds_shade_the_ground_and_part_around_a_pillar() {
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

    // Sand with its top at y = 0, and a 3×3 pillar from there up to 110,
    // through the clouds at 96 to 100.
    let bounds = WorldBounds {
        min_y: -32,
        max_y: 127,
    };
    let mut world = World::new();
    for y in -1..=3 {
        for z in -2..=2 {
            for x in -2..=2 {
                let fill = if y < 0 { sand } else { BlockId::AIR };
                world.insert_chunk(ChunkPos::new(x, y, z), Chunk::filled(fill));
            }
        }
    }
    for y in 0..110 {
        for z in 15..18 {
            for x in 15..18 {
                world.set_block(BlockPos::new(x, y, z), stone);
            }
        }
    }
    let mut engine = LightEngine::new(content.blocks(), bounds);
    for y in (-1..=3).rev() {
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
    // Whether there's a cloud over block (x, z) of cloud space, and the
    // blocks around it.
    let cloudy = |x: i32, z: i32| {
        [(-4, -4), (4, -4), (-4, 4), (4, 4)]
            .iter()
            .all(|&(dx, dz)| {
                let (x, z) = (
                    (x + dx).div_euclid(CLOUD_CELL),
                    (z + dz).div_euclid(CLOUD_CELL),
                );
                cloud_at(SKY.seed, SKY.cover, x, z)
            })
    };

    let scene = |camera: Camera, clouds: Option<CloudSky>| Scene {
        camera,
        target: None,
        view_distance: 160.0,
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

    // Under the clouds, looking down at the sand away from the pillar.
    let mut below = Camera::new(DVec3::new(-30.0, 30.0, -30.0));
    below.rotate(0.0, -1.5);
    let ground = Vec3::new(-30.0, 0.0, -35.0);
    // The noon sun is a little south of overhead: the cloud that shades the
    // ground is that far north of it.
    assert!(cloudy(-30, -35 + 24));
    let clear = pixel(
        &mut renderer,
        &scene(below, None),
        ground,
        "clouds-none.png",
    );
    let shaded = pixel(
        &mut renderer,
        &scene(below, Some(SKY)),
        ground,
        "clouds-shade.png",
    );
    assert!(
        brightness(shaded) < brightness(clear) * 0.85,
        "clouds darken the ground: {shaded:?} against {clear:?}"
    );

    // Above the clouds, looking down next to the pillar.
    let mut above = Camera::new(DVec3::new(12.0, 140.0, 12.0));
    above.rotate(0.0, -1.5);
    let beside = Vec3::new(10.0, 100.0, 10.0);
    let far = Vec3::new(44.0, 100.0, 12.0);
    assert!(cloudy(10, 10) && cloudy(43, 12));
    let unparted = pixel(
        &mut renderer,
        &scene(above, Some(SKY)),
        beside,
        "clouds-unparted.png",
    );
    for &pos in &positions {
        let chunk = world.chunk(pos).unwrap();
        let columns = cloud_obstacles(pos, chunk, |block| content.blocks().is_solid(block));
        renderer.set_cloud_obstacles(pos, columns);
    }
    // Obstacles count once some time has passed; nudge the clock.
    let later = CloudSky { time: 1.0, ..SKY };
    let parted = pixel(
        &mut renderer,
        &scene(above, Some(later)),
        beside,
        "clouds-parted.png",
    );
    let still_far = pixel(
        &mut renderer,
        &scene(above, Some(later)),
        far,
        "clouds-parted.png",
    );
    let difference = |a: [f32; 3], b: [f32; 3]| (0..3).map(|i| (a[i] - b[i]).abs()).sum::<f32>();
    assert!(
        difference(parted, unparted) > 60.0,
        "clouds part beside the pillar: {parted:?} against {unparted:?}"
    );
    assert!(
        difference(still_far, unparted) < 30.0,
        "and stay away from it: {still_far:?} against {unparted:?}"
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
