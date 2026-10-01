//! Finding the chunks that can be seen from the camera: those in the view
//! frustum that a line of sight reaches through open blocks.
//!
//! A breadth-first walk from the camera's chunk steps into a neighbour only
//! through faces that the current chunk connects (see [`Visibility`]), and
//! never back towards the camera. Solid ground stops it, so caves under the
//! player's feet and the far sides of mountains are skipped.

use std::collections::{HashSet, VecDeque};
use std::ops::RangeInclusive;

use glam::{DVec3, Vec3};
use ruda_core::{BlockPos, CHUNK_SIZE, ChunkPos, Face};

use crate::{Frustum, Visibility};

/// Chunks the camera may see, nearest first. `visibility` tells how a loaded
/// chunk lets sight through; chunks it doesn't know stop the walk. Above the
/// world (`world_y` holds the chunk heights inside it) there is only air.
pub(crate) fn visible_chunks(
    camera: DVec3,
    frustum: &Frustum,
    radius: i32,
    world_y: RangeInclusive<i32>,
    visibility: impl Fn(ChunkPos) -> Option<Visibility>,
    out: &mut Vec<ChunkPos>,
) {
    out.clear();
    let start = BlockPos(camera.floor().as_ivec3()).chunk();
    // Air above the world, as high as needed to look down on it.
    let sky_top = start.0.y.max(*world_y.end()) + 1;
    let lookup = |pos: ChunkPos| {
        let y = pos.0.y;
        if y > *world_y.end() {
            (y <= sky_top).then_some(Visibility::ALL)
        } else if y < *world_y.start() {
            None
        } else {
            visibility(pos)
        }
    };

    let mut visited = HashSet::new();
    let mut queue = VecDeque::new();
    visited.insert(start);
    // (chunk, the face it was entered through, directions travelled so far)
    queue.push_back((start, None::<Face>, 0u8));
    while let Some((pos, entered, travelled)) = queue.pop_front() {
        out.push(pos);
        let inside = lookup(pos);
        for face in Face::ALL {
            if travelled & (1 << face.opposite().index()) != 0 {
                continue;
            }
            if let Some(entered) = entered
                && !inside.is_some_and(|v| v.connects(entered, face))
            {
                continue;
            }
            let next = pos.offset(face);
            let offset = next.0 - start.0;
            if offset.x * offset.x + offset.z * offset.z > (radius + 1) * (radius + 1)
                || visited.contains(&next)
                || lookup(next).is_none()
            {
                continue;
            }
            let min = (next.origin().0.as_dvec3() - camera).as_vec3();
            if !frustum.intersects_box(min, min + Vec3::splat(CHUNK_SIZE as f32)) {
                continue;
            }
            visited.insert(next);
            queue.push_back((next, Some(face.opposite()), travelled | 1 << face.index()));
        }
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use super::*;

    /// A world of air chunks above y = 0, a surface layer at y = 0 that only
    /// opens upwards and sideways, and solid stone below with one cave at
    /// (3, −1, 0) that connects to nothing.
    fn layered(pos: ChunkPos) -> Option<Visibility> {
        let y = pos.0.y;
        Some(match y {
            1.. => Visibility::ALL,
            0 => surface(),
            _ if pos == ChunkPos::new(3, -1, 0) => Visibility::ALL,
            _ => Visibility::NONE,
        })
    }

    fn surface() -> Visibility {
        // Open from the sides and the top to each other, but not to the
        // bottom: what the walk would compute for a chunk of ground with
        // some air above it.
        let open = (1 << Face::PosX.index())
            | (1 << Face::NegX.index())
            | (1 << Face::PosY.index())
            | (1 << Face::PosZ.index())
            | (1 << Face::NegZ.index());
        let mut faces = [open; 6];
        faces[Face::NegY.index()] = 0;
        Visibility::from_faces(faces)
    }

    fn run(camera: DVec3, world: impl Fn(ChunkPos) -> Option<Visibility>) -> Vec<ChunkPos> {
        let mut out = Vec::new();
        visible_chunks(camera, &Frustum::everything(), 4, -4..=3, world, &mut out);
        out
    }

    #[test]
    fn ground_hides_what_is_below_it() {
        let seen = run(DVec3::new(16.0, 40.0, 16.0), layered);
        assert_eq!(seen[0], ChunkPos::new(0, 1, 0));
        assert!(seen.contains(&ChunkPos::new(0, 0, 0)));
        assert!(seen.contains(&ChunkPos::new(4, 0, 0)));
        assert!(!seen.iter().any(|pos| pos.0.y < 0));
    }

    #[test]
    fn underground_the_cave_sees_only_itself() {
        let seen = run(DVec3::new(3.0 * 32.0 + 16.0, -16.0, 16.0), layered);
        // Out of the cave every way leads into stone, which is seen but not
        // looked through.
        assert_eq!(seen[0], ChunkPos::new(3, -1, 0));
        assert_eq!(seen.len(), 7);
    }

    #[test]
    fn unknown_chunks_stop_the_walk_and_the_sky_is_open() {
        let mut known = HashMap::new();
        for x in -1..=1 {
            known.insert(ChunkPos::new(x, 3, 0), Visibility::ALL);
        }
        // Above the top of the world (chunk y = 3), looking down.
        let seen = run(DVec3::new(16.0, 200.0, 16.0), |pos| {
            known.get(&pos).copied()
        });
        assert!(seen.contains(&ChunkPos::new(1, 3, 0)));
        assert!(!seen.contains(&ChunkPos::new(2, 3, 0)));
        assert!(!seen.iter().any(|pos| pos.0.y < 3));
    }
}
