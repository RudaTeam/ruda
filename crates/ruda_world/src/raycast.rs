use glam::{DVec3, IVec3};
use ruda_core::{BlockPos, Face};

/// Where a ray stopped.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct RayHit {
    pub block: BlockPos,
    /// The face of `block` the ray entered through; a block placed against
    /// the hit goes to `block.offset(face)`.
    pub face: Face,
    /// Distance from the ray origin to the entry point.
    pub distance: f64,
}

/// Walks the blocks a ray passes through, in order, and returns the first one
/// for which `is_solid` holds, at most `max_distance` away. The block that
/// contains the origin is skipped.
///
/// This is the voxel traversal of Amanatides and Woo: at every step the ray
/// crosses whichever block boundary it reaches first.
pub fn raycast(
    origin: DVec3,
    direction: DVec3,
    max_distance: f64,
    mut is_solid: impl FnMut(BlockPos) -> bool,
) -> Option<RayHit> {
    let direction = direction.normalize_or_zero();
    if direction == DVec3::ZERO {
        return None;
    }

    let mut block = origin.floor().as_ivec3();
    let step = IVec3::new(sign(direction.x), sign(direction.y), sign(direction.z));
    // Ray length needed to cross one whole block along each axis.
    let t_delta = direction.recip().abs();
    // Ray length to the first boundary along each axis.
    let mut t_max = DVec3::from_array(std::array::from_fn(|axis| match step[axis] {
        1 => (f64::from(block[axis]) + 1.0 - origin[axis]) * t_delta[axis],
        -1 => (origin[axis] - f64::from(block[axis])) * t_delta[axis],
        _ => f64::INFINITY,
    }));

    loop {
        let axis = t_max.min_position();
        let distance = t_max[axis];
        if distance > max_distance {
            return None;
        }
        block[axis] += step[axis];
        t_max[axis] += t_delta[axis];

        let pos = BlockPos(block);
        if is_solid(pos) {
            // Moving towards +axis the ray enters through the block's
            // negative face, and the other way round.
            let face = Face::ALL[axis * 2 + usize::from(step[axis] > 0)];
            return Some(RayHit {
                block: pos,
                face,
                distance,
            });
        }
    }
}

fn sign(value: f64) -> i32 {
    if value > 0.0 {
        1
    } else if value < 0.0 {
        -1
    } else {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn floor_at_y0(pos: BlockPos) -> bool {
        pos.0.y <= 0
    }

    #[test]
    fn looking_down_hits_the_top_face() {
        let hit = raycast(DVec3::new(0.5, 10.5, 0.5), DVec3::NEG_Y, 20.0, floor_at_y0).unwrap();
        assert_eq!(hit.block, BlockPos::new(0, 0, 0));
        assert_eq!(hit.face, Face::PosY);
        assert!((hit.distance - 9.5).abs() < 1e-9);
        assert_eq!(hit.block.offset(hit.face), BlockPos::new(0, 1, 0));
    }

    #[test]
    fn hits_walls_from_either_side() {
        let wall_at_x5 = |pos: BlockPos| pos.0.x == 5;
        let hit = raycast(DVec3::new(0.5, 0.5, 0.5), DVec3::X, 10.0, wall_at_x5).unwrap();
        assert_eq!((hit.block, hit.face), (BlockPos::new(5, 0, 0), Face::NegX));

        let hit = raycast(DVec3::new(9.5, 0.5, 0.5), DVec3::NEG_X, 10.0, wall_at_x5).unwrap();
        assert_eq!((hit.block, hit.face), (BlockPos::new(5, 0, 0), Face::PosX));
        assert!((hit.distance - 3.5).abs() < 1e-9);
    }

    #[test]
    fn stops_at_max_distance() {
        assert_eq!(
            raycast(DVec3::new(0.5, 10.5, 0.5), DVec3::NEG_Y, 5.0, floor_at_y0),
            None
        );
        assert_eq!(raycast(DVec3::ZERO, DVec3::ZERO, 5.0, floor_at_y0), None);
    }

    #[test]
    fn handles_negative_coordinates_and_diagonals() {
        let target = BlockPos::new(-3, -2, -4);
        let origin = DVec3::new(-0.5, 0.25, -0.75);
        let direction = target.0.as_dvec3() + 0.5 - origin;
        let hit = raycast(origin, direction, 10.0, |pos| pos == target).unwrap();
        assert_eq!(hit.block, target);
    }
}
