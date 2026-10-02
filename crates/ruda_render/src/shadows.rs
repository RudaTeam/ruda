//! Sun shadows: cascades of the shadow map, squares around the camera
//! looking from the sun (or the moon), each wider and coarser than the one
//! before. The last reaches as far as the far-away look of the world, and
//! looks far towards the sun: mountains cast their long shadows across
//! valleys at sunset.

use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{DVec3, Mat4, Vec3};

/// Edge length of each cascade of the shadow map, in texels.
pub(crate) const SHADOW_MAP_SIZE: u32 = 2048;
pub(crate) const CASCADES: usize = 4;
/// How far from the camera each cascade but the last reaches, if the view
/// does.
const REACH: [f32; CASCADES - 1] = [16.0, 48.0, 160.0];
/// Room towards the light (and away from it) for things outside a cascade
/// that still cast shadows into it, like a mountain behind the camera: for
/// the near cascades, and for the last one.
pub(crate) const CASTER_MARGIN: f32 = 256.0;
pub(crate) const FAR_CASTER_MARGIN: f32 = 2048.0;

/// How far each cascade reaches for a view of `view_distance` blocks, with
/// far-away terrain out to `far_distance`. The last cascade reaches no
/// farther than the one before when there is nothing farther to shade.
pub(crate) fn reach(view_distance: f32, far_distance: f32) -> [f32; CASCADES] {
    let near = view_distance.clamp(8.0, REACH[2]);
    [
        REACH[0].min(near * 0.25),
        REACH[1].min(near * 0.5),
        near,
        view_distance.max(far_distance).max(near),
    ]
}

/// The margin of cascade `index`; see [`CASTER_MARGIN`].
pub(crate) fn margin(index: usize) -> f32 {
    if index == CASCADES - 1 {
        FAR_CASTER_MARGIN
    } else {
        CASTER_MARGIN
    }
}

/// A light view of the square `radius` blocks around `camera`, lit from
/// `toward_light`, relative to the camera, seeing `margin` blocks farther
/// towards the light.
///
/// The square doesn't turn with the camera, and it moves in whole texels,
/// so shadow edges don't shimmer as the camera moves or turns. Its "up" is
/// the axis the sun and moon turn around (they cross the sky in a circle
/// around +z), so as they move, the texels stay put along that axis.
pub(crate) fn cascade(camera: DVec3, radius: f32, toward_light: Vec3, margin: f32) -> Mat4 {
    let rotation =
        glam::dcamera::rh::view::look_to_mat4(DVec3::ZERO, -toward_light.as_dvec3(), DVec3::Z);
    let texel = f64::from(2.0 * radius / SHADOW_MAP_SIZE as f32);
    let in_light = rotation.transform_point3(camera);
    let snapped = DVec3::new(
        (in_light.x / texel).floor() * texel,
        (in_light.y / texel).floor() * texel,
        in_light.z,
    );
    let center = (rotation.inverse().transform_point3(snapped) - camera).as_vec3();

    let eye = center + toward_light * (radius + margin);
    let view = look_to_mat4(eye, -toward_light, Vec3::Z);
    let projection = directx::orthographic(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        2.0 * (radius + margin),
    );
    projection * view
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_cascade_covers_the_square_around_the_camera() {
        let camera = DVec3::new(1000.5, 70.0, -300.25);
        let sun = Vec3::new(0.4, 0.8, 0.25).normalize();
        let matrix = cascade(camera, 48.0, sun, CASTER_MARGIN);
        for point in [
            Vec3::ZERO,
            Vec3::new(40.0, -10.0, 0.0),
            Vec3::new(0.0, 5.0, -40.0),
        ] {
            let p = matrix.project_point3(point);
            assert!(p.x.abs() < 1.0 && p.y.abs() < 1.0, "{p}");
            assert!((0.0..1.0).contains(&p.z), "{p}");
        }
        // Higher up towards the sun is closer to it.
        let low = matrix.project_point3(Vec3::ZERO);
        let high = matrix.project_point3(sun * 5.0);
        assert!(high.z < low.z);
    }

    #[test]
    fn texels_stay_put_as_the_camera_moves() {
        let sun = Vec3::new(0.4, 0.8, 0.25).normalize();
        let a = DVec3::new(10.0, 70.0, 10.0);
        let b = a + DVec3::new(3.3, 0.7, -1.9);
        // The same block corner, seen from either camera, lands on the same
        // spot of the texel grid.
        let corner = DVec3::new(20.0, 64.0, 15.0);
        let texel_of = |camera: DVec3| {
            let p = cascade(camera, 48.0, sun, CASTER_MARGIN)
                .project_point3((corner - camera).as_vec3());
            (p.truncate() * SHADOW_MAP_SIZE as f32 * 0.5).fract()
        };
        let (fa, fb) = (texel_of(a), texel_of(b));
        let wrapped = |d: f32| (d - d.round()).abs();
        assert!(
            wrapped(fa.x - fb.x) < 0.01 && wrapped(fa.y - fb.y) < 0.01,
            "{fa} {fb}"
        );
    }

    #[test]
    fn short_views_keep_the_cascades_in_order() {
        let r = reach(48.0, 0.0);
        assert!(r[0] < r[1] && r[1] < r[2] && r[2] == 48.0, "{r:?}");
        assert_eq!(r[3], r[2], "nothing farther to shade");
        assert_eq!(reach(320.0, 2048.0), [16.0, 48.0, 160.0, 2048.0]);
        assert_eq!(reach(512.0, 0.0)[3], 512.0);
    }
}
