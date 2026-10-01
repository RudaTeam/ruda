//! The air sunlight passes through. How air molecules (Rayleigh) and haze
//! (Mie) scatter light, and how ozone absorbs it, decides how much of the sun
//! reaches the ground and what colour the sky is in every direction. Sky,
//! sunlight, fog and clouds all take their colours from here, so a sunset
//! reddens all of them together.
//!
//! Everything is worked out once, at start-up, for every height of the light
//! above the horizon; a frame only looks the numbers up. The method follows
//! Hillaire's "A Scalable and Production Ready Sky and Atmosphere Rendering
//! Technique" (2020): a table of how much light gets through the air, one of
//! light scattered many times, and from them the sky seen from the ground.
//!
//! Distances are in kilometres. Light is in units where a white surface
//! facing the sun outside the air looks 1: the sun's illuminance is π.

use std::f64::consts::{FRAC_PI_2, PI, TAU};
use std::sync::OnceLock;

use glam::{DVec3, Vec3};
use rayon::prelude::*;

/// Edge sizes of the sky table: directions around from the light, heights
/// above the horizon, and heights of the light.
pub(crate) const SKY_AZIMUTHS: u32 = 32;
pub(crate) const SKY_ELEVATIONS: u32 = 32;
pub(crate) const SKY_LIGHTS: u32 = 48;
/// The light this far below the horizon (degrees) no longer lights the sky.
const LOWEST_LIGHT: f64 = -20.0;

const GROUND: f64 = 6360.0;
const TOP: f64 = 6460.0;
/// Height of the eye above the ground: the world's few hundred blocks don't
/// matter next to a hundred kilometres of air.
const EYE: f64 = 0.3;
const SUN: f64 = PI;
const RAYLEIGH: DVec3 = DVec3::new(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_HEIGHT: f64 = 8.0;
const MIE_SCATTERING: f64 = 3.996e-3;
const MIE_EXTINCTION: f64 = 4.44e-3;
const MIE_HEIGHT: f64 = 1.2;
/// How strongly haze scatters forwards: the bright glow around the sun.
const MIE_G: f64 = 0.8;
const OZONE: DVec3 = DVec3::new(0.650e-3, 1.881e-3, 0.085e-3);
const OZONE_PEAK: f64 = 25.0;
const OZONE_HALF_WIDTH: f64 = 15.0;
/// How much light the land reflects back up into the air.
const GROUND_ALBEDO: f64 = 0.3;

/// Sizes of the helper tables.
const TRANSMITTANCE_SIZE: (usize, usize) = (256, 64);
const MULTIPLE_SIZE: usize = 32;

/// The sky worked out for every height of the light.
pub(crate) struct SkyTables {
    transmittance: Grid,
    /// Sky colour by direction, per height of the light: `SKY_AZIMUTHS` ×
    /// `SKY_ELEVATIONS` × `SKY_LIGHTS` texels of four half floats, as a 3-D
    /// texture wants them.
    pub sky: Vec<u16>,
    /// Sky light on a surface, per height of the light.
    ambient: Vec<Ambient>,
}

/// The light of the whole sky on a surface, divided by π: for a surface
/// facing `n`, `constant + toward * n_toward + up * n.y`, where `n_toward`
/// is how much the surface faces the light's side of the sky. These are the
/// first two orders of spherical harmonics.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Ambient {
    constant: DVec3,
    toward: DVec3,
    up: DVec3,
}

/// What a light (the sun or the moon) at some height does to the scene.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Lit {
    /// What a white surface facing the light reflects of its direct light.
    pub direct: Vec3,
    /// Sky light on a surface by its normal, divided by π: constant, x, y
    /// and z terms.
    pub ambient: [Vec3; 4],
    /// Texture coordinate of this light's layer in the sky table; `None`
    /// when the light is too far below the horizon to matter.
    pub layer: Option<f32>,
}

impl SkyTables {
    /// The tables, worked out on first use: a few tens of milliseconds.
    pub(crate) fn get() -> &'static Self {
        static TABLES: OnceLock<SkyTables> = OnceLock::new();
        TABLES.get_or_init(Self::compute)
    }

    fn compute() -> Self {
        let transmittance = Grid::build(TRANSMITTANCE_SIZE.0, TRANSMITTANCE_SIZE.1, |u, v| {
            let (r, mu) = transmittance_r_mu(u, v);
            (-optical_depth(r, mu)).exp()
        });
        let air = Air {
            transmittance,
            multiple: Grid::default(),
        };
        let multiple = Grid::build(MULTIPLE_SIZE, MULTIPLE_SIZE, |u, v| {
            air.multiple_scattering(GROUND + v * (TOP - GROUND), u * 2.0 - 1.0)
        });
        let air = Air { multiple, ..air };

        let (width, height) = (SKY_AZIMUTHS as usize, SKY_ELEVATIONS as usize);
        let layers: Vec<(Grid, Ambient)> = (0..SKY_LIGHTS)
            .into_par_iter()
            .map(|layer| {
                let height_of_light = light_angle(f64::from(layer) / f64::from(SKY_LIGHTS - 1));
                let grid = Grid::build_serial(width, height, |u, v| {
                    air.sky(height_of_light, u * u * PI, v * v * FRAC_PI_2)
                });
                let ambient = air.ambient(&grid, height_of_light);
                (grid, ambient)
            })
            .collect();
        let mut sky = Vec::with_capacity(width * height * layers.len() * 4);
        for (grid, _) in &layers {
            for texel in &grid.texels {
                sky.extend([texel.x, texel.y, texel.z, 1.0].map(|c| f16_bits(c as f32)));
            }
        }
        Self {
            transmittance: air.transmittance,
            sky,
            ambient: layers.into_iter().map(|(_, ambient)| ambient).collect(),
        }
    }

    /// The light from `toward` (a unit vector) on the scene, `strength`
    /// times as bright as the sun.
    pub(crate) fn light(&self, toward: Vec3, strength: f32) -> Lit {
        let toward = toward.as_dvec3();
        let angle = toward.y.clamp(-1.0, 1.0).asin().to_degrees();
        let Some(slice) = light_slice(angle) else {
            return Lit {
                direct: Vec3::ZERO,
                ambient: [Vec3::ZERO; 4],
                layer: None,
            };
        };
        let last = (SKY_LIGHTS - 1) as usize;
        let below = (slice.floor() as usize).min(last);
        let above = (below + 1).min(last);
        let t = slice - below as f64;
        let [a, b] = [below, above].map(|i| self.ambient[i]);
        let mix = |a: DVec3, b: DVec3| a.lerp(b, t);
        let ambient = Ambient {
            constant: mix(a.constant, b.constant),
            toward: mix(a.toward, b.toward),
            up: mix(a.up, b.up),
        };
        // The light's side of the sky, flat.
        let side = DVec3::new(toward.x, 0.0, toward.z).normalize_or_zero();
        let strength = f64::from(strength);
        let to_f32 = |v: DVec3| (v * strength).as_vec3();
        Lit {
            direct: to_f32(self.direct(toward.y)),
            ambient: [
                to_f32(ambient.constant),
                to_f32(ambient.toward * side.x),
                to_f32(ambient.up),
                to_f32(ambient.toward * side.z),
            ],
            layer: Some(((slice + 0.5) / f64::from(SKY_LIGHTS)) as f32),
        }
    }

    /// What a white surface facing a light at height `mu` (the sine of its
    /// angle above the horizon) reflects of its direct light. It fades over
    /// the light's own width as it sinks below the horizon.
    fn direct(&self, mu: f64) -> DVec3 {
        let r = GROUND + EYE;
        let horizon = horizon_mu(r);
        let fade = smoothstep(horizon - 0.01, horizon + 0.01, mu);
        if fade == 0.0 {
            return DVec3::ZERO;
        }
        let mu = mu.max(horizon + 1e-4);
        self.transmittance.sample(transmittance_uv(r, mu)) * fade * SUN / PI
    }
}

/// The air, with the tables that are ready so far.
struct Air {
    transmittance: Grid,
    multiple: Grid,
}

impl Air {
    /// How much light gets through the air from height `r` along `mu` (the
    /// cosine from straight up), or none if the ground is in the way.
    fn transmittance(&self, r: f64, mu: f64) -> DVec3 {
        if hits_ground(r, mu) {
            DVec3::ZERO
        } else {
            self.transmittance.sample(transmittance_uv(r, mu))
        }
    }

    /// Light scattered twice and more at height `r` with the sun at `mu`,
    /// per unit of sunlight, as if it were scattered evenly in all
    /// directions.
    fn multiple_scattering(&self, r: f64, mu_s: f64) -> DVec3 {
        const DIRECTIONS: usize = 64;
        const STEPS: usize = 20;
        let sun = DVec3::new((1.0 - mu_s * mu_s).max(0.0).sqrt(), mu_s, 0.0);
        let origin = DVec3::new(0.0, r, 0.0);
        let isotropic = 1.0 / (4.0 * PI);
        let mut light = DVec3::ZERO;
        let mut transfer = DVec3::ZERO;
        for direction in fibonacci_sphere(DIRECTIONS) {
            let (length, ground) = ray_length(origin, direction);
            let dt = length / STEPS as f64;
            let mut through = DVec3::ONE;
            for i in 0..STEPS {
                let p = origin + direction * ((i as f64 + 0.5) * dt);
                let pr = p.length();
                let air = medium(pr - GROUND);
                let scattering = air.rayleigh + DVec3::splat(air.mie);
                let step = (-air.extinction * dt).exp();
                let lit = self.transmittance(pr, p.dot(sun) / pr);
                let source = scattering * lit * isotropic;
                light += through * integrate(source, step, air.extinction, dt);
                transfer += through * integrate(scattering, step, air.extinction, dt);
                through *= step;
            }
            if ground {
                let p = (origin + direction * length).normalize();
                let facing = p.dot(sun).max(0.0);
                light += through * self.transmittance(GROUND, facing) * facing * GROUND_ALBEDO / PI;
            }
        }
        light /= DIRECTIONS as f64;
        transfer /= DIRECTIONS as f64;
        light / (DVec3::ONE - transfer)
    }

    /// The sky seen from the eye with the light `light` radians above the
    /// horizon, looking `azimuth` radians around from it and `elevation`
    /// radians up.
    fn sky(&self, light: f64, azimuth: f64, elevation: f64) -> DVec3 {
        const STEPS: usize = 30;
        let origin = DVec3::new(0.0, GROUND + EYE, 0.0);
        let view = DVec3::new(
            elevation.cos() * azimuth.cos(),
            elevation.sin(),
            elevation.cos() * azimuth.sin(),
        );
        let sun = DVec3::new(light.cos(), light.sin(), 0.0);
        let cos_theta = view.dot(sun);
        let (rayleigh_phase, mie_phase) = (rayleigh_phase(cos_theta), mie_phase(cos_theta));
        let (length, _) = ray_length(origin, view);
        let mut light = DVec3::ZERO;
        let mut through = DVec3::ONE;
        for i in 0..STEPS {
            // Short steps near the eye, where the air is thickest.
            let t0 = length * (i as f64 / STEPS as f64).powi(2);
            let t1 = length * ((i + 1) as f64 / STEPS as f64).powi(2);
            let dt = t1 - t0;
            let p = origin + view * ((t0 + t1) * 0.5);
            let pr = p.length();
            let mu_s = p.dot(sun) / pr;
            let air = medium(pr - GROUND);
            let lit = self.transmittance(pr, mu_s);
            let multiple = self.multiple.sample(DVec3::new(
                (mu_s + 1.0) * 0.5,
                (pr - GROUND) / (TOP - GROUND),
                0.0,
            ));
            let source = (air.rayleigh * rayleigh_phase + DVec3::splat(air.mie * mie_phase)) * lit
                + multiple * (air.rayleigh + DVec3::splat(air.mie));
            let step = (-air.extinction * dt).exp();
            light += through * integrate(source, step, air.extinction, dt);
            through *= step;
        }
        light * SUN
    }

    /// Light of the whole sky on surfaces, from one layer of the sky table:
    /// the table above the horizon, lit land below it.
    fn ambient(&self, sky: &Grid, light: f64) -> Ambient {
        const AZIMUTHS: usize = 64;
        const ELEVATIONS: usize = 24;
        let (d_azimuth, d_elevation) = (TAU / AZIMUTHS as f64, FRAC_PI_2 / ELEVATIONS as f64);
        let mut sum = DVec3::ZERO;
        let (mut along_x, mut along_y) = (DVec3::ZERO, DVec3::ZERO);
        for j in 0..ELEVATIONS {
            let elevation = (j as f64 + 0.5) * d_elevation;
            for i in 0..AZIMUTHS {
                let azimuth = (i as f64 + 0.5) * d_azimuth;
                let folded = if azimuth > PI { TAU - azimuth } else { azimuth };
                let radiance = sky.sample(DVec3::new(
                    (folded / PI).sqrt(),
                    (elevation / FRAC_PI_2).sqrt(),
                    0.0,
                ));
                let solid_angle = elevation.cos() * d_elevation * d_azimuth;
                sum += radiance * solid_angle;
                along_x += radiance * (elevation.cos() * azimuth.cos() * solid_angle);
                along_y += radiance * (elevation.sin() * solid_angle);
            }
        }
        // The land, lit by the sun and by the sky above it.
        let up = along_y / PI;
        let sun = self.direct_on_land(light);
        let land = (sun * light.sin().max(0.0) + up) * GROUND_ALBEDO;
        sum += land * TAU;
        along_y -= land * PI;
        Ambient {
            constant: sum / (4.0 * PI),
            toward: along_x / TAU,
            up: along_y / TAU,
        }
    }

    fn direct_on_land(&self, light: f64) -> DVec3 {
        let r = GROUND + EYE;
        self.transmittance(r, light.sin()) * SUN / PI
    }
}

/// Scattering and extinction per kilometre at a height above the ground.
struct Medium {
    rayleigh: DVec3,
    mie: f64,
    extinction: DVec3,
}

fn medium(height: f64) -> Medium {
    let height = height.max(0.0);
    let rayleigh = RAYLEIGH * (-height / RAYLEIGH_HEIGHT).exp();
    let haze = (-height / MIE_HEIGHT).exp();
    let ozone = (1.0 - (height - OZONE_PEAK).abs() / OZONE_HALF_WIDTH).max(0.0);
    Medium {
        rayleigh,
        mie: MIE_SCATTERING * haze,
        extinction: rayleigh + DVec3::splat(MIE_EXTINCTION * haze) + OZONE * ozone,
    }
}

/// Light added over a step of length `dt` by `source` per kilometre, as the
/// air along the step also dims it: `step` is the step's transmittance.
fn integrate(source: DVec3, step: DVec3, extinction: DVec3, dt: f64) -> DVec3 {
    DVec3::new(
        integrate_channel(source.x, step.x, extinction.x, dt),
        integrate_channel(source.y, step.y, extinction.y, dt),
        integrate_channel(source.z, step.z, extinction.z, dt),
    )
}

fn integrate_channel(source: f64, step: f64, extinction: f64, dt: f64) -> f64 {
    if extinction > 1e-12 {
        source * (1.0 - step) / extinction
    } else {
        source * dt
    }
}

fn rayleigh_phase(cos_theta: f64) -> f64 {
    3.0 / (16.0 * PI) * (1.0 + cos_theta * cos_theta)
}

/// The Cornette-Shanks phase function.
fn mie_phase(cos_theta: f64) -> f64 {
    let g2 = MIE_G * MIE_G;
    3.0 / (8.0 * PI) * (1.0 - g2) * (1.0 + cos_theta * cos_theta)
        / ((2.0 + g2) * (1.0 + g2 - 2.0 * MIE_G * cos_theta).powf(1.5))
}

/// Optical depth from height `r` along `mu` to the top of the air.
fn optical_depth(r: f64, mu: f64) -> DVec3 {
    const STEPS: usize = 40;
    let origin = DVec3::new(0.0, r, 0.0);
    let direction = DVec3::new((1.0 - mu * mu).max(0.0).sqrt(), mu, 0.0);
    let length = sphere_exit(origin, direction, TOP);
    let dt = length / STEPS as f64;
    (0..STEPS)
        .map(|i| {
            let p = origin + direction * ((i as f64 + 0.5) * dt);
            medium(p.length() - GROUND).extinction * dt
        })
        .sum()
}

/// How far a ray goes through the air, and whether it ends on the ground.
fn ray_length(origin: DVec3, direction: DVec3) -> (f64, bool) {
    let r = origin.length();
    let mu = origin.dot(direction) / r;
    if hits_ground(r, mu) {
        let b = origin.dot(direction);
        let c = origin.length_squared() - GROUND * GROUND;
        (-b - (b * b - c).max(0.0).sqrt(), true)
    } else {
        (sphere_exit(origin, direction, TOP), false)
    }
}

/// Distance along a ray from inside a sphere around the planet's centre to
/// where it leaves it.
fn sphere_exit(origin: DVec3, direction: DVec3, radius: f64) -> f64 {
    let b = origin.dot(direction);
    let c = origin.length_squared() - radius * radius;
    -b + (b * b - c).max(0.0).sqrt()
}

fn hits_ground(r: f64, mu: f64) -> bool {
    mu < 0.0 && r * r * (mu * mu - 1.0) + GROUND * GROUND >= 0.0
}

/// The height (cosine from straight up) of the horizon seen from `r`.
fn horizon_mu(r: f64) -> f64 {
    -(1.0 - (GROUND / r).powi(2)).max(0.0).sqrt()
}

/// Table coordinates for the transmittance from `r` along `mu`, for rays
/// that miss the ground (after Bruneton): fine near the horizon.
fn transmittance_uv(r: f64, mu: f64) -> DVec3 {
    let h = (TOP * TOP - GROUND * GROUND).sqrt();
    let rho = (r * r - GROUND * GROUND).max(0.0).sqrt();
    let discriminant = r * r * (mu * mu - 1.0) + TOP * TOP;
    let d = (-r * mu + discriminant.max(0.0).sqrt()).max(0.0);
    let (d_min, d_max) = (TOP - r, rho + h);
    DVec3::new((d - d_min) / (d_max - d_min), rho / h, 0.0)
}

fn transmittance_r_mu(u: f64, v: f64) -> (f64, f64) {
    let h = (TOP * TOP - GROUND * GROUND).sqrt();
    let rho = h * v;
    let r = (rho * rho + GROUND * GROUND).sqrt();
    let (d_min, d_max) = (TOP - r, rho + h);
    let d = d_min + u * (d_max - d_min);
    let mu = if d == 0.0 {
        1.0
    } else {
        ((h * h - rho * rho - d * d) / (2.0 * r * d)).clamp(-1.0, 1.0)
    };
    (r, mu)
}

/// Heights of the light are spread more thickly near the horizon, where
/// the sky changes fastest: the table coordinate goes with the square root
/// of the angle.
fn spread(degrees: f64) -> f64 {
    degrees.signum() * (degrees.abs() / 90.0).sqrt()
}

/// The angle of the light, in radians, at a fraction of the table.
fn light_angle(fraction: f64) -> f64 {
    let (low, high) = (spread(LOWEST_LIGHT), spread(90.0));
    let s = low + fraction * (high - low);
    (s.signum() * s * s * 90.0).to_radians()
}

/// Which layer of the table (fractional) holds a light `degrees` above the
/// horizon, if any.
fn light_slice(degrees: f64) -> Option<f64> {
    if degrees < LOWEST_LIGHT {
        return None;
    }
    let (low, high) = (spread(LOWEST_LIGHT), spread(90.0));
    Some((spread(degrees.min(90.0)) - low) / (high - low) * f64::from(SKY_LIGHTS - 1))
}

/// Evenly spread unit vectors.
fn fibonacci_sphere(count: usize) -> impl Iterator<Item = DVec3> {
    let golden = PI * (3.0 - 5f64.sqrt());
    (0..count).map(move |i| {
        let y = 1.0 - (i as f64 + 0.5) / count as f64 * 2.0;
        let radius = (1.0 - y * y).sqrt();
        let angle = golden * i as f64;
        DVec3::new(angle.cos() * radius, y, angle.sin() * radius)
    })
}

fn smoothstep(edge0: f64, edge1: f64, x: f64) -> f64 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

/// A table of colours over the unit square, read with linear filtering
/// between texel centres, the way a GPU reads a texture.
#[derive(Default)]
struct Grid {
    width: usize,
    height: usize,
    texels: Vec<DVec3>,
}

impl Grid {
    /// Fills the table in parallel; `texel` gets coordinates from 0 to 1 at
    /// the first and last texel centres.
    fn build(width: usize, height: usize, texel: impl Fn(f64, f64) -> DVec3 + Sync) -> Self {
        let texels = (0..width * height)
            .into_par_iter()
            .map(|i| texel(fraction(i % width, width), fraction(i / width, height)))
            .collect();
        Self {
            width,
            height,
            texels,
        }
    }

    fn build_serial(width: usize, height: usize, texel: impl Fn(f64, f64) -> DVec3) -> Self {
        let texels = (0..width * height)
            .map(|i| texel(fraction(i % width, width), fraction(i / width, height)))
            .collect();
        Self {
            width,
            height,
            texels,
        }
    }

    /// The colour at `uv.x`, `uv.y` (`z` is ignored), each from 0 to 1.
    fn sample(&self, uv: DVec3) -> DVec3 {
        let x = uv.x.clamp(0.0, 1.0) * (self.width - 1) as f64;
        let y = uv.y.clamp(0.0, 1.0) * (self.height - 1) as f64;
        let (x0, y0) = (x.floor() as usize, y.floor() as usize);
        let (x1, y1) = ((x0 + 1).min(self.width - 1), (y0 + 1).min(self.height - 1));
        let (tx, ty) = (x - x0 as f64, y - y0 as f64);
        let at = |x: usize, y: usize| self.texels[y * self.width + x];
        let near = at(x0, y0).lerp(at(x1, y0), tx);
        let far = at(x0, y1).lerp(at(x1, y1), tx);
        near.lerp(far, ty)
    }
}

fn fraction(index: usize, size: usize) -> f64 {
    index as f64 / (size - 1) as f64
}

/// The nearest half float to a finite, non-negative `value`.
fn f16_bits(value: f32) -> u16 {
    let bits = value.max(0.0).to_bits();
    let exponent = ((bits >> 23) & 0xff) as i32 - 127 + 15;
    let mantissa = bits & 0x7f_ffff;
    if exponent >= 31 {
        return 0x7bff;
    }
    if exponent <= 0 {
        if exponent < -10 {
            return 0;
        }
        let mantissa = mantissa | 0x80_0000;
        let shift = (14 - exponent) as u32;
        let half = mantissa >> shift;
        let round = (mantissa >> (shift - 1)) & 1;
        return (half + round) as u16;
    }
    let half = ((exponent as u32) << 10) | (mantissa >> 13);
    // Rounding may carry into the exponent, which is right.
    (half + ((mantissa >> 12) & 1)) as u16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn half_floats_round_to_nearest() {
        assert_eq!(f16_bits(0.0), 0);
        assert_eq!(f16_bits(1.0), 0x3c00);
        assert_eq!(f16_bits(0.5), 0x3800);
        assert_eq!(f16_bits(65504.0), 0x7bff);
        assert_eq!(f16_bits(1e9), 0x7bff);
        assert_eq!(f16_bits(1.0 / 3.0), 0x3555);
        // The smallest subnormal.
        assert_eq!(f16_bits(5.960_464_5e-8), 1);
    }

    #[test]
    fn transmittance_coordinates_round_trip() {
        for (u, v) in [(0.1, 0.2), (0.5, 0.5), (0.9, 0.05)] {
            let (r, mu) = transmittance_r_mu(u, v);
            let uv = transmittance_uv(r, mu);
            assert!(
                (uv.x - u).abs() < 1e-6 && (uv.y - v).abs() < 1e-6,
                "{u} {v} → {uv}"
            );
        }
    }

    #[test]
    fn light_slices_cover_the_table() {
        assert_eq!(light_slice(LOWEST_LIGHT), Some(0.0));
        let top = light_slice(90.0).unwrap();
        assert!((top - f64::from(SKY_LIGHTS - 1)).abs() < 1e-9);
        assert_eq!(light_slice(-30.0), None);
        for degrees in [-15.0, -2.0, 0.0, 3.0, 40.0] {
            let slice = light_slice(degrees).unwrap();
            let back = light_angle(slice / f64::from(SKY_LIGHTS - 1)).to_degrees();
            assert!((back - degrees).abs() < 1e-6, "{degrees} → {back}");
        }
        // More layers near the horizon than high up.
        let low = light_slice(5.0).unwrap() - light_slice(0.0).unwrap();
        let high = light_slice(85.0).unwrap() - light_slice(80.0).unwrap();
        assert!(low > high * 2.0);
    }

    #[test]
    fn the_noon_sky_is_blue_and_the_setting_sun_red() {
        let tables = SkyTables::get();
        let noon = tables.light(Vec3::new(0.0, 1.0, 0.0), 1.0);
        let sunset = tables.light(Vec3::new(1.0, 0.02, 0.0).normalize(), 1.0);
        // White, strong light at noon; weak, red light at sunset.
        assert!(noon.direct.min_element() > 0.7, "{}", noon.direct);
        assert!(noon.direct.x / noon.direct.z < 1.3, "{}", noon.direct);
        assert!(sunset.direct.x > sunset.direct.z * 3.0, "{}", sunset.direct);
        assert!(sunset.direct.x < noon.direct.x * 0.5, "{}", sunset.direct);
        // Sky light from above is blue at noon.
        let up = noon.ambient[0] + noon.ambient[2];
        assert!(up.z > up.x, "{up}");
        assert!(up.x > 0.02 && up.z < 0.6, "{up}");
        // Surfaces facing the setting sun get more sky light than those
        // facing away.
        let toward = sunset.ambient[0] + sunset.ambient[1];
        let away = sunset.ambient[0] - sunset.ambient[1];
        assert!(toward.x > away.x, "{toward} {away}");
        // Deep night: nothing.
        let night = tables.light(Vec3::new(0.0, -1.0, 0.0), 1.0);
        assert_eq!(night.layer, None);
        assert_eq!(night.direct, Vec3::ZERO);
    }

    #[test]
    fn the_sky_table_is_complete() {
        let tables = SkyTables::get();
        let texels = (SKY_AZIMUTHS * SKY_ELEVATIONS * SKY_LIGHTS) as usize;
        assert_eq!(tables.sky.len(), texels * 4);
        // Nothing is infinite or not a number.
        assert!(tables.sky.iter().all(|&bits| bits & 0x7c00 != 0x7c00));
    }
}
