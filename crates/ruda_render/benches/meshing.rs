//! How long it takes to turn generated terrain into geometry.
//!
//! `cargo bench -p ruda_render --bench meshing`

use std::hint::black_box;

use criterion::{Criterion, criterion_group, criterion_main};
use ruda_core::{ChunkPos, ContentBuilder, WorldBounds};
use ruda_render::{BlockFaces, PaddedChunk, mesh_chunk};
use ruda_world::World;
use ruda_world::light::LightEngine;

/// A 4×4 patch of chunks at height `y` with all their neighbours loaded.
fn terrain(y: i32) -> (World, Vec<ChunkPos>, BlockFaces) {
    let mut content = ContentBuilder::new();
    ruda_base::register(&mut content).unwrap();
    let content = content.build();
    let generator = ruda_base::terrain(content.blocks(), 2024, WorldBounds::DEFAULT).unwrap();
    let mut world = World::new();
    for cy in y - 1..=y + 1 {
        for cz in -1..=4 {
            for cx in -1..=4 {
                let pos = ChunkPos::new(cx, cy, cz);
                world.insert_chunk(pos, generator.generate(pos));
            }
        }
    }
    let patch = (0..4)
        .flat_map(|z| (0..4).map(move |x| ChunkPos::new(x, y, z)))
        .collect();
    // One layer per block is enough: texture layers don't affect the work.
    let faces = BlockFaces::new(content.blocks(), |_| 1);
    (world, patch, faces)
}

fn meshing(c: &mut Criterion) {
    for (name, y) in [("surface", 0), ("underground", -4)] {
        let (world, patch, faces) = terrain(y);
        let chunks: Vec<PaddedChunk> = patch
            .iter()
            .map(|&pos| PaddedChunk::gather(&world, pos).unwrap())
            .collect();
        let quads: usize = chunks
            .iter()
            .map(|chunk| mesh_chunk(chunk, &faces).quads.len())
            .sum();
        println!(
            "{name}: {} quads per chunk on average, {} KiB of geometry",
            quads / chunks.len(),
            quads * 16 / chunks.len() / 1024
        );

        let mut next = chunks.iter().cycle();
        c.bench_function(&format!("mesh {name} chunk"), |b| {
            b.iter(|| mesh_chunk(black_box(next.next().unwrap()), &faces))
        });
        let mut next = patch.iter().cycle();
        c.bench_function(&format!("gather {name} chunk"), |b| {
            b.iter(|| PaddedChunk::gather(black_box(&world), *next.next().unwrap()))
        });
    }
}

/// Lighting a 3×3 patch of whole chunk columns, top down, as the server
/// does when the world loads.
fn lighting(c: &mut Criterion) {
    let mut content = ContentBuilder::new();
    ruda_base::register(&mut content).unwrap();
    let content = content.build();
    let bounds = WorldBounds::DEFAULT;
    let generator = ruda_base::terrain(content.blocks(), 2024, bounds).unwrap();
    let mut world = World::new();
    let mut order = Vec::new();
    for y in ((bounds.min_y >> 5)..=(bounds.max_y >> 5)).rev() {
        for z in -1..=1 {
            for x in -1..=1 {
                let pos = ChunkPos::new(x, y, z);
                world.insert_chunk(pos, generator.generate(pos));
                order.push(pos);
            }
        }
    }
    println!("lighting: {} chunks per run", order.len());
    c.bench_function("light 3x3 columns", |b| {
        b.iter_batched(
            || world.clone(),
            |mut world| {
                let mut engine = LightEngine::new(content.blocks(), bounds);
                for &pos in &order {
                    engine.light_chunk(&mut world, pos);
                }
                world
            },
            criterion::BatchSize::LargeInput,
        )
    });
}

criterion_group!(benches, meshing, lighting);
criterion_main!(benches);
