//! Draws a view of a generated world into a PNG, without a window: for
//! looking at how the world is lit and drawn at a given place and time.
//!
//! ```sh
//! cargo run --release -p ruda_render --example view -- \
//!     --seed 7 --camera 420,118,20,225,-12 --time 2000 --out view.png
//! ```
//!
//! Options: `--seed N`, `--camera X,Y,Z[,YAW,PITCH]` (degrees), `--time
//! TICKS` (0 sunrise, 6000 noon), `--size WxH`, `--radius CHUNKS` of the
//! world generated around the camera, `--lod BLOCKS` of far terrain beyond
//! it, `--shadows`, `--no-clouds`, `--exposure X` instead of following
//! how bright the view is, `--out PATH`.

use std::path::PathBuf;

use anyhow::{Context, Result, bail};
use glam::{DVec3, IVec3};
use ruda_core::{BlockPos, ChunkPos, ContentBuilder, Light, WorldBounds};
use ruda_render::{
    Backdrop, Camera, CloudSky, PaddedChunk, Renderer, Scene, cloud_obstacles, far_cloud_obstacles,
    mesh_chunk, mesh_lod,
};
use ruda_world::World;
use ruda_world::light::LightEngine;
use ruda_world::lod::{LOD_TILE_SIZE, LodTile, LodTilePos};

const TICK_RATE: f64 = 20.0;
const DAY_LENGTH: f64 = 24_000.0;

struct Options {
    seed: u64,
    camera: Camera,
    time: f64,
    size: (u32, u32),
    radius: i32,
    lod: f32,
    shadows: bool,
    clouds: bool,
    exposure: Option<f32>,
    out: PathBuf,
}

fn options() -> Result<Options> {
    let mut options = Options {
        seed: 7,
        camera: Camera::new(DVec3::new(0.0, 80.0, 0.0)),
        time: 2000.0,
        size: (1280, 720),
        radius: 6,
        lod: 1024.0,
        shadows: false,
        clouds: true,
        exposure: None,
        out: PathBuf::from("view.png"),
    };
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        let mut value = || args.next().with_context(|| format!("{arg} needs a value"));
        match arg.as_str() {
            "--seed" => options.seed = value()?.parse()?,
            "--time" => options.time = value()?.parse()?,
            "--radius" => options.radius = value()?.parse()?,
            "--lod" => options.lod = value()?.parse()?,
            "--out" => options.out = value()?.into(),
            "--shadows" => options.shadows = true,
            "--exposure" => options.exposure = Some(value()?.parse()?),
            "--no-clouds" => options.clouds = false,
            "--size" => {
                let value = value()?;
                let (width, height) = value.split_once('x').context("--size WxH")?;
                options.size = (width.parse()?, height.parse()?);
            }
            "--camera" => {
                let numbers: Vec<f64> = value()?
                    .split(',')
                    .map(str::parse)
                    .collect::<Result<_, _>>()?;
                let [x, y, z, ref rest @ ..] = numbers[..] else {
                    bail!("--camera X,Y,Z[,YAW,PITCH]");
                };
                let mut camera = Camera::new(DVec3::new(x, y, z));
                if let &[yaw, pitch] = rest {
                    camera.rotate(yaw.to_radians() as f32, pitch.to_radians() as f32);
                }
                options.camera = camera;
            }
            _ => bail!("unknown option {arg}"),
        }
    }
    Ok(options)
}

fn main() -> Result<()> {
    let options = options()?;
    let mut renderer = pollster::block_on(Renderer::headless(options.size.0, options.size.1))?;
    renderer.set_shadows(options.shadows);
    renderer.fix_exposure(options.exposure);
    let mut content = ContentBuilder::new();
    ruda_base::register(&mut content)?;
    let content = content.build();
    let blocks = content.blocks();
    let bounds = WorldBounds::DEFAULT;
    let generator = ruda_base::terrain(blocks, options.seed, bounds)?;
    let faces = renderer.load_block_textures(&content);
    renderer.set_sky_textures(ruda_base::SUN, ruda_base::MOON);

    // Whole columns around the camera, lit from the top down.
    let eye = options.camera.position.floor().as_ivec3();
    let center = BlockPos::new(eye.x, eye.y, eye.z).chunk();
    let layers = (bounds.min_y >> 5)..=(bounds.max_y >> 5);
    let mut world = World::new();
    let columns: Vec<(i32, i32)> = (-options.radius..=options.radius)
        .flat_map(|z| (-options.radius..=options.radius).map(move |x| (x, z)))
        .filter(|(x, z)| x * x + z * z <= options.radius * options.radius)
        .map(|(x, z)| (center.0.x + x, center.0.z + z))
        .collect();
    for &(x, z) in &columns {
        for y in layers.clone() {
            let pos = ChunkPos(IVec3::new(x, y, z));
            world.insert_chunk(pos, generator.generate(pos));
        }
    }
    let mut light = LightEngine::new(blocks, bounds);
    for y in layers.clone().rev() {
        for &(x, z) in &columns {
            light.light_chunk(&mut world, ChunkPos(IVec3::new(x, y, z)));
        }
    }
    for &(x, z) in &columns {
        for y in layers.clone() {
            let pos = ChunkPos(IVec3::new(x, y, z));
            if let Some(chunk) = PaddedChunk::gather(&world, pos) {
                renderer.upload_chunk(pos, &mesh_chunk(&chunk, &faces));
            }
            let chunk = world.chunk(pos).context("generated")?;
            renderer.set_cloud_obstacles(pos, cloud_obstacles(pos, chunk, |b| blocks.is_solid(b)));
        }
    }

    // Far terrain, in tiles out to the LOD distance.
    let reach = (options.lod / LOD_TILE_SIZE as f32).ceil() as i32 + 1;
    let camera = options.camera.position;
    let home = LodTilePos::containing(camera.x as i32, camera.z as i32);
    for z in -reach..=reach {
        for x in -reach..=reach {
            let pos = LodTilePos::new(home.x + x, home.z + z);
            if let Some(tile) = LodTile::generate(&generator, pos) {
                renderer.upload_lod(pos, &mesh_lod(&tile, &faces));
                renderer.set_far_cloud_obstacles(pos, Some(far_cloud_obstacles(&tile)));
            }
        }
    }

    let scene = Scene {
        camera: options.camera,
        target: None,
        view_distance: (options.radius * 32) as f32,
        bounds: Some(bounds),
        time_of_day: (options.time.rem_euclid(DAY_LENGTH) / DAY_LENGTH) as f32,
        // Views from under the open sky only.
        eye_light: Light::SKY,
        lod_distance: options.lod,
        clouds: options.clouds.then(|| CloudSky {
            seed: options
                .seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(29),
            cover: 0.35,
            time: options.time / TICK_RATE,
        }),
    };
    let (width, height, pixels) = renderer.capture(Backdrop::World(&scene), None)?;
    let file = std::io::BufWriter::new(std::fs::File::create(&options.out)?);
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(&pixels)?;
    println!("{}", options.out.display());
    Ok(())
}
