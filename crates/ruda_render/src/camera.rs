use glam::camera::rh::{proj::directx, view::look_to_mat4};
use glam::{DVec3, Mat4, Vec3, Vec4};

/// A first-person camera. Rendering happens relative to its position, so
/// precision does not degrade far from the world origin.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Camera {
    pub position: DVec3,
    /// Turn around the vertical axis in radians; 0 looks towards −Z and
    /// positive values turn right.
    pub yaw: f32,
    /// Look up (positive) or down, in radians.
    pub pitch: f32,
    /// Vertical field of view in radians.
    pub fov_y: f32,
}

/// Just short of straight up or down, where the view would flip.
const MAX_PITCH: f32 = 89.0_f32.to_radians();

impl Camera {
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            yaw: 0.0,
            pitch: 0.0,
            fov_y: 70.0_f32.to_radians(),
        }
    }

    pub fn rotate(&mut self, yaw: f32, pitch: f32) {
        self.yaw = (self.yaw + yaw).rem_euclid(std::f32::consts::TAU);
        self.pitch = (self.pitch + pitch).clamp(-MAX_PITCH, MAX_PITCH);
    }

    pub fn forward(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        let (sin_pitch, cos_pitch) = self.pitch.sin_cos();
        Vec3::new(sin_yaw * cos_pitch, sin_pitch, -cos_yaw * cos_pitch)
    }

    /// Horizontal right direction.
    pub fn right(&self) -> Vec3 {
        let (sin_yaw, cos_yaw) = self.yaw.sin_cos();
        Vec3::new(cos_yaw, 0.0, sin_yaw)
    }

    /// Projection times view for a camera at the origin.
    pub fn view_proj(&self, aspect: f32, far: f32) -> Mat4 {
        // Depth from 0 to 1, as in WebGPU.
        let projection = directx::perspective(self.fov_y, aspect, 0.05, far);
        projection * look_to_mat4(Vec3::ZERO, self.forward(), Vec3::Y)
    }
}

/// The six planes of a view frustum, for culling boxes outside the view.
#[derive(Clone, Copy, Debug)]
pub struct Frustum {
    planes: [Vec4; 6],
}

impl Frustum {
    pub fn new(view_proj: Mat4) -> Self {
        let [r0, r1, r2, r3] = [0, 1, 2, 3].map(|row| view_proj.row(row));
        // Clip space z runs from 0 to 1, so the near plane is just r2.
        Self {
            planes: [r3 + r0, r3 - r0, r3 + r1, r3 - r1, r2, r3 - r2],
        }
    }

    /// Whether an axis-aligned box may be visible.
    pub fn intersects_box(&self, min: Vec3, max: Vec3) -> bool {
        self.planes.iter().all(|plane| {
            let normal = plane.truncate();
            // The box corner furthest along the plane normal.
            let corner = Vec3::select(normal.cmpge(Vec3::ZERO), max, min);
            normal.dot(corner) + plane.w >= 0.0
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directions_follow_yaw_and_pitch() {
        let mut camera = Camera::new(DVec3::ZERO);
        assert!(camera.forward().abs_diff_eq(Vec3::NEG_Z, 1e-6));
        assert!(camera.right().abs_diff_eq(Vec3::X, 1e-6));
        camera.rotate(std::f32::consts::FRAC_PI_2, 0.0);
        assert!(camera.forward().abs_diff_eq(Vec3::X, 1e-6));
        camera.rotate(0.0, 10.0);
        assert!(camera.pitch < std::f32::consts::FRAC_PI_2);
    }

    #[test]
    fn culls_boxes_behind_the_camera() {
        let camera = Camera::new(DVec3::ZERO);
        let frustum = Frustum::new(camera.view_proj(16.0 / 9.0, 500.0));
        let ahead = Vec3::new(0.0, 0.0, -50.0);
        assert!(frustum.intersects_box(ahead - 1.0, ahead + 1.0));
        let behind = Vec3::new(0.0, 0.0, 50.0);
        assert!(!frustum.intersects_box(behind - 1.0, behind + 1.0));
        let beyond = Vec3::new(0.0, 0.0, -900.0);
        assert!(!frustum.intersects_box(beyond - 1.0, beyond + 1.0));
    }
}
