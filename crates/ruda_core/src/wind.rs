//! Wind: which way and how fast the air moves over the world. Clouds drift
//! with it; weather, rain and smoke will too.

use glam::DVec2;

/// The wind over the world, in blocks per second along x (east) and z
/// (south).
///
/// For now it blows the same everywhere and never changes. Weather will make
/// it differ from place to place and over time; whoever asks for the wind
/// at a point, or for how far it has carried the air, won't need to change.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WindMap {
    steady: DVec2,
}

impl WindMap {
    /// A light breeze, mostly east.
    pub const BREEZE: Self = Self::steady(DVec2::new(1.0, 0.35));

    /// The same wind everywhere, always.
    pub const fn steady(velocity: DVec2) -> Self {
        Self { steady: velocity }
    }

    /// The wind over block column `(x, z)`.
    pub fn at(&self, _x: f64, _z: f64) -> DVec2 {
        self.steady
    }

    /// How far the wind has carried the air `seconds` after the world
    /// began. Clouds are drawn this far downwind of where they began, so
    /// every player sees them in the same place.
    pub fn drift(&self, seconds: f64) -> DVec2 {
        self.steady * seconds
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_steady_wind_carries_the_air_evenly() {
        let wind = WindMap::steady(DVec2::new(2.0, -1.0));
        assert_eq!(wind.at(10.0, -5000.0), DVec2::new(2.0, -1.0));
        assert_eq!(wind.drift(0.0), DVec2::ZERO);
        assert_eq!(wind.drift(30.0), DVec2::new(60.0, -30.0));
    }
}
