//! Culling may only skip what can't be seen: from points in the open, the
//! world drawn with culling looks exactly like the world drawn without it.
//!
//! Needs a GPU; without one the test only says so and passes.

use glam::DVec3;
use ruda_core::{BlockId, BlockPos, ChunkPos, ContentBuilder, Face, WorldBounds};
use ruda_render::{Backdrop, Camera, PaddedChunk, Renderer, Scene, mesh_chunk};
use ruda_world::World;

const RADIUS: i32 = 2;

#[test]
fn culling_hides_nothing_that_can_be_seen() {
    let mut renderer = match pollster::block_on(Renderer::headless(480, 270)) {
        Ok(renderer) => renderer,
        Err(error) => {
            eprintln!("skipping, no GPU: {error:#}");
            return;
        }
    };
    let mut content = ContentBuilder::new();
    ruda_base::register(&mut content).unwrap();
    let content = content.build();
    let bounds = WorldBounds::DEFAULT;
    let generator = ruda_base::terrain(content.blocks(), 7, bounds).unwrap();

    let mut world = World::new();
    for y in (bounds.min_y >> 5)..=(bounds.max_y >> 5) {
        for z in -RADIUS..=RADIUS {
            for x in -RADIUS..=RADIUS {
                let pos = ChunkPos::new(x, y, z);
                world.insert_chunk(pos, generator.generate(pos));
            }
        }
    }
    let faces = renderer.load_block_textures(&content);
    let positions: Vec<ChunkPos> = world.chunks().map(|(pos, _)| pos).collect();
    for pos in positions {
        let chunk = PaddedChunk::gather(&world, pos).unwrap();
        renderer.upload_chunk(pos, &mesh_chunk(&chunk, &faces));
    }

    let ground = f64::from(generator.surface_height(0, 0));
    let mut cameras = vec![
        // Standing on the ground, looking a little down.
        (DVec3::new(0.5, ground + 2.0, 0.5), 0.0, -0.3),
        // Flying high, looking down at an angle.
        (DVec3::new(0.5, ground + 40.0, 0.5), 1.0, -1.0),
        // Above the top of the world.
        (DVec3::new(30.0, 190.0, -20.0), 2.5, -1.4),
    ];
    let cave = find_cave(&world).expect("the test area has a cave");
    cameras.push((cave, 0.7, -0.2));
    cameras.push((cave, 3.5, 0.4));

    let mut culled_somewhere = false;
    for (position, yaw, pitch) in cameras {
        let mut camera = Camera::new(position);
        camera.rotate(yaw, pitch);
        let scene = Scene {
            camera,
            target: None,
            view_distance: (RADIUS * 32) as f32,
            bounds: Some(bounds),
        };
        renderer.set_culling(true);
        let (_, _, culled) = renderer.capture(Backdrop::World(&scene), None).unwrap();
        let culled_quads = renderer.stats().quads;
        renderer.set_culling(false);
        let (_, _, all) = renderer.capture(Backdrop::World(&scene), None).unwrap();
        let all_quads = renderer.stats().quads;

        let different = culled
            .chunks(4)
            .zip(all.chunks(4))
            .filter(|(a, b)| a.iter().zip(b.iter()).any(|(a, b)| a.abs_diff(*b) > 2))
            .count();
        assert_eq!(
            different, 0,
            "culling changed {different} pixels seen from {position}"
        );
        culled_somewhere |= culled_quads < all_quads / 2;
    }
    assert!(
        culled_somewhere,
        "culling should skip a good part of the world somewhere"
    );
}

/// An open block underground, in the middle of the test area, with open
/// blocks all around it.
fn find_cave(world: &World) -> Option<DVec3> {
    let air = |pos: BlockPos| world.block(pos) == Some(BlockId::AIR);
    (-64..-8).rev().find_map(|y| {
        (-20..20).find_map(|z| {
            (-20..20).find_map(|x| {
                let pos = BlockPos::new(x, y, z);
                let open = air(pos) && Face::ALL.iter().all(|&face| air(pos.offset(face)));
                open.then(|| pos.0.as_dvec3() + 0.5)
            })
        })
    })
}
