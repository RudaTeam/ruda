//! Sun shadows: the three cascades of the shadow map, each looking from the
//! sun (or the moon) at the part of the view it covers, from sharp near the
//! camera to coarse far away.

use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{DVec3, Mat4, Vec3};

use crate::Camera;

/// Edge length of each cascade of the shadow map, in texels.
pub(crate) const SHADOW_MAP_SIZE: u32 = 2048;
/// How many cascades there are.
pub(crate) const CASCADES: usize = 3;
/// Distances where the first two cascades end. Each next one starts a
/// fifth before, and the two blend over that stretch.
const SPLITS: [f32; 2] = [16.0, 56.0];
/// Shadows end here, or at the view distance if that is shorter.
const SHADOW_DISTANCE: f32 = 160.0;
/// Room above and below a cascade for things outside the view that still
/// cast shadows into it, like a mountain behind the camera.
const CASTER_MARGIN: f32 = 256.0;

#[derive(Clone, Copy, Debug)]
pub(crate) struct Cascades {
    /// Camera-relative position to shadow map space, near cascade first.
    pub view_proj: [Mat4; CASCADES],
    /// Where the first two cascades end.
    pub splits: [f32; 2],
    /// Where shadows end.
    pub end: f32,
    /// How wide a texel of each cascade is, in blocks.
    pub texel: [f32; CASCADES],
}

/// Fits the cascades around the view of `camera`, lit from `toward_light`.
pub(crate) fn cascades(
    camera: &Camera,
    aspect: f32,
    view_distance: f32,
    toward_light: Vec3,
) -> Cascades {
    let end = view_distance.min(SHADOW_DISTANCE);
    let splits = SPLITS.map(|split| split.min(end));
    let ranges = [
        (0.05, splits[0]),
        (splits[0] * 0.8, splits[1]),
        (splits[1] * 0.8, end),
    ];
    let fits = ranges.map(|(near, far)| fit(camera, aspect, near, far, toward_light));
    Cascades {
        view_proj: fits.map(|(view_proj, _)| view_proj),
        splits,
        end,
        texel: fits.map(|(_, radius)| 2.0 * radius / SHADOW_MAP_SIZE as f32),
    }
}

/// A light view of the slice of the view from `near` to `far`: a cube around
/// the sphere that holds the slice, so its size doesn't change as the camera
/// turns, and moved in whole texels, so shadow edges don't shimmer. With
/// the sphere's radius.
fn fit(camera: &Camera, aspect: f32, near: f32, far: f32, toward_light: Vec3) -> (Mat4, f32) {
    let forward = camera.forward();
    let right = camera.right();
    let up = right.cross(forward);
    let tan_y = (camera.fov_y * 0.5).tan();
    let tan_x = tan_y * aspect;
    let mut corners = Vec::with_capacity(8);
    for distance in [near, far] {
        for sx in [-1.0, 1.0] {
            for sy in [-1.0, 1.0] {
                corners.push(
                    forward * distance
                        + right * (sx * distance * tan_x)
                        + up * (sy * distance * tan_y),
                );
            }
        }
    }
    let center = corners.iter().copied().sum::<Vec3>() / corners.len() as f32;
    let radius = corners
        .iter()
        .map(|corner| corner.distance(center))
        .fold(0.0, f32::max);
    let radius = (radius * 16.0).ceil() / 16.0;

    let up_hint = if toward_light.y.abs() > 0.99 {
        DVec3::Z
    } else {
        DVec3::Y
    };
    let rotation =
        glam::dcamera::rh::view::look_to_mat4(DVec3::ZERO, -toward_light.as_dvec3(), up_hint);
    let texel = f64::from(2.0 * radius / SHADOW_MAP_SIZE as f32);
    let in_light = rotation.transform_point3(camera.position + center.as_dvec3());
    let snapped = DVec3::new(
        (in_light.x / texel).floor() * texel,
        (in_light.y / texel).floor() * texel,
        in_light.z,
    );
    let center = (rotation.inverse().transform_point3(snapped) - camera.position).as_vec3();

    let eye = center + toward_light * (radius + CASTER_MARGIN);
    let view = look_to_mat4(eye, -toward_light, up_hint.as_vec3());
    let projection = directx::orthographic(
        -radius,
        radius,
        -radius,
        radius,
        0.0,
        2.0 * (radius + CASTER_MARGIN),
    );
    (projection * view, radius)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cascades_cover_what_the_camera_sees() {
        let mut camera = Camera::new(DVec3::new(1000.5, 70.0, -300.25));
        camera.rotate(0.7, -0.2);
        let sun = Vec3::new(0.4, 0.8, 0.2).normalize();
        let cascades = cascades(&camera, 16.0 / 9.0, 128.0, sun);
        assert_eq!(cascades.end, 128.0);
        // Farther cascades are coarser.
        assert!(cascades.texel[0] < cascades.texel[1] && cascades.texel[1] < cascades.texel[2]);
        // Points straight ahead land inside the map, in front of the light.
        for (cascade, distance) in [(0, 10.0), (1, 40.0), (2, 100.0)] {
            let point = camera.forward() * distance;
            let p = cascades.view_proj[cascade].project_point3(point);
            assert!(p.x.abs() < 1.0 && p.y.abs() < 1.0, "{p}");
            assert!((0.0..1.0).contains(&p.z), "{p}");
        }
        // Higher up towards the sun is closer to it.
        let low = cascades.view_proj[0].project_point3(camera.forward() * 10.0);
        let high = cascades.view_proj[0].project_point3(camera.forward() * 10.0 + sun * 5.0);
        assert!(high.z < low.z);
    }
}
