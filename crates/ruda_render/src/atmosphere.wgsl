// The air: how sunlight is scattered and absorbed on its way through it,
// after "A Scalable and Production Ready Sky and Atmosphere Rendering
// Technique" (Hillaire, 2020), with Earth's air. Distances are in
// kilometres from the planet's centre; a block is a metre.
//
// Only functions and constants: `sky_tables.wgsl` and `world.wgsl` are each
// joined to this file.

const PLANET = 6360.0;
const TOP = 6460.0;
// How high the world's sea level is above the planet's surface.
const SEA_LEVEL = 0.2;
const RAYLEIGH_SCATTERING = vec3<f32>(5.802e-3, 13.558e-3, 33.1e-3);
const RAYLEIGH_HEIGHT = 8.0;
const MIE_SCATTERING = 3.996e-3;
const MIE_EXTINCTION = 4.40e-3;
const MIE_HEIGHT = 1.2;
const MIE_G = 0.8;
const OZONE_ABSORPTION = vec3<f32>(0.650e-3, 1.881e-3, 0.085e-3);
const GROUND_ALBEDO = vec3<f32>(0.3);
const PI = 3.14159265;
// The sun's light above the air, in the frame's units of light: a white
// surface facing it in full sun shows at about this divided by π.
const SUN_LIGHT = vec3<f32>(10.0, 9.8, 9.5);
// The moon's light as a share of the sun's: far more than in life, so
// nights are dark but not black. Moonlight is sunlight, but eyes see night
// blue.
const MOON_LIGHT = 0.03;
const MOON_TINT = vec3<f32>(0.55, 0.7, 1.0);

// What a kilometre of air at `height` above the surface scatters and how
// much light it takes away.
struct Medium {
    rayleigh: vec3<f32>,
    mie: f32,
    extinction: vec3<f32>,
}

fn medium(height: f32) -> Medium {
    let rayleigh = exp(-max(height, 0.0) / RAYLEIGH_HEIGHT);
    let mie = exp(-max(height, 0.0) / MIE_HEIGHT);
    let ozone = max(0.0, 1.0 - abs(height - 25.0) / 15.0);
    var out: Medium;
    out.rayleigh = RAYLEIGH_SCATTERING * rayleigh;
    out.mie = MIE_SCATTERING * mie;
    out.extinction = out.rayleigh + MIE_EXTINCTION * mie + OZONE_ABSORPTION * ozone;
    return out;
}

// How far along `direction` from `origin` a sphere of `radius` around the
// planet's centre is: the nearest crossing ahead, or -1 if none.
fn ray_sphere(origin: vec3<f32>, direction: vec3<f32>, radius: f32) -> f32 {
    let b = dot(origin, direction);
    let c = dot(origin, origin) - radius * radius;
    let d = b * b - c;
    if d < 0.0 {
        return -1.0;
    }
    let root = sqrt(d);
    let near = -b - root;
    let far = -b + root;
    if near > 0.0 {
        return near;
    }
    if far > 0.0 {
        return far;
    }
    return -1.0;
}

fn rayleigh_phase(cos_angle: f32) -> f32 {
    return 3.0 / (16.0 * PI) * (1.0 + cos_angle * cos_angle);
}

fn mie_phase(cos_angle: f32) -> f32 {
    // Cornette-Shanks.
    let g2 = MIE_G * MIE_G;
    let k = 3.0 / (8.0 * PI) * (1.0 - g2) / (2.0 + g2);
    return k * (1.0 + cos_angle * cos_angle)
        / pow(max(1.0 + g2 - 2.0 * MIE_G * cos_angle, 1e-4), 1.5);
}

// Where the transmittance table holds the way from radius `r` at the cosine
// `mu` of the angle from straight up, after Bruneton.
fn transmittance_uv(r: f32, mu: f32) -> vec2<f32> {
    let h = sqrt(TOP * TOP - PLANET * PLANET);
    let rho = sqrt(max(r * r - PLANET * PLANET, 0.0));
    let discriminant = r * r * (mu * mu - 1.0) + TOP * TOP;
    let d = max(0.0, -r * mu + sqrt(max(discriminant, 0.0)));
    let d_min = TOP - r;
    let d_max = rho + h;
    return vec2<f32>((d - d_min) / (d_max - d_min), rho / h);
}

// The other way: the radius and cosine a texel of the table stands for.
fn transmittance_r_mu(uv: vec2<f32>) -> vec2<f32> {
    let h = sqrt(TOP * TOP - PLANET * PLANET);
    let rho = h * uv.y;
    let r = sqrt(rho * rho + PLANET * PLANET);
    let d_min = TOP - r;
    let d_max = rho + h;
    let d = d_min + uv.x * (d_max - d_min);
    var mu = 1.0;
    if d > 0.0 {
        mu = clamp((h * h - rho * rho - d * d) / (2.0 * r * d), -1.0, 1.0);
    }
    return vec2<f32>(r, mu);
}

// Where the multiple scattering table holds radius `r` and the cosine of
// the sun's angle from straight up.
fn multiple_uv(r: f32, sun_mu: f32) -> vec2<f32> {
    return vec2<f32>(sun_mu * 0.5 + 0.5, clamp((r - PLANET) / (TOP - PLANET), 0.0, 1.0));
}

// Where the sky table holds a direction, seen from radius `r`: the cosine
// of its angle from straight up, and of its angle around from the sun's
// side. Most texels go to near the horizon, where the sky changes most.
fn sky_uv(r: f32, view_mu: f32, sun_side: f32) -> vec2<f32> {
    let horizon = sqrt(max(r * r - PLANET * PLANET, 0.0));
    let beta = acos(clamp(horizon / r, -1.0, 1.0));
    let zenith_horizon = PI - beta;
    let angle = acos(clamp(view_mu, -1.0, 1.0));
    var v: f32;
    if angle < zenith_horizon {
        v = (1.0 - sqrt(max(1.0 - angle / zenith_horizon, 0.0))) * 0.5;
    } else {
        v = sqrt(max((angle - zenith_horizon) / beta, 0.0)) * 0.5 + 0.5;
    }
    return vec2<f32>(sqrt(clamp(-sun_side * 0.5 + 0.5, 0.0, 1.0)), v);
}

// The cosine of the angle around from the sun's side for `direction`: 1
// straight towards the sun, -1 away.
fn sun_side(direction: vec3<f32>, sun: vec3<f32>) -> f32 {
    let flat_direction = direction.xz;
    let flat_sun = sun.xz;
    let lengths = length(flat_direction) * length(flat_sun);
    if lengths < 1e-5 {
        return 1.0;
    }
    return dot(flat_direction, flat_sun) / lengths;
}

// The planet radius of a point `y` blocks above sea level.
fn radius_at(y: f32) -> f32 {
    return PLANET + SEA_LEVEL + y * 0.001;
}

// Whether looking from radius `r` at the cosine `mu` from straight up runs
// into the ground.
fn below_horizon(r: f32, mu: f32) -> bool {
    let direction = vec3<f32>(sqrt(max(1.0 - mu * mu, 0.0)), mu, 0.0);
    return ray_sphere(vec3<f32>(0.0, r, 0.0), direction, PLANET) > 0.0;
}
