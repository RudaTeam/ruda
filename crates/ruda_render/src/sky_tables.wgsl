// Tables of how the air lights the sky, joined to `atmosphere.wgsl`: what
// reaches a point from the sun (transmittance) and from light scattered
// many times, worked out once; the sky in every direction from the camera,
// and the light the whole sky sheds on the ground, worked out every frame.

struct Tables {
    // xyz: towards the sun; w: the camera's height above sea level, in
    // blocks.
    sun: vec4<f32>,
}

@group(0) @binding(0) var<uniform> tables: Tables;
@group(0) @binding(1) var table_sampler: sampler;
@group(0) @binding(2) var transmittance: texture_2d<f32>;
@group(0) @binding(3) var multiple: texture_2d<f32>;
@group(0) @binding(4) var sky_view: texture_2d<f32>;

struct Fullscreen {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> Fullscreen {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Fullscreen;
    out.clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

// The share of light that gets from radius `r` out of the air at cosine
// `mu` from straight up; none below the horizon.
fn sun_transmittance(r: f32, mu: f32) -> vec3<f32> {
    if below_horizon(r, mu) {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(transmittance, table_sampler, transmittance_uv(r, mu), 0.0).rgb;
}

@fragment
fn transmittance_table(in: Fullscreen) -> @location(0) vec4<f32> {
    let r_mu = transmittance_r_mu(in.uv);
    let origin = vec3<f32>(0.0, r_mu.x, 0.0);
    let direction = vec3<f32>(sqrt(max(1.0 - r_mu.y * r_mu.y, 0.0)), r_mu.y, 0.0);
    let span = max(ray_sphere(origin, direction, TOP), 0.0);
    let steps = 40;
    var depth = vec3<f32>(0.0);
    for (var i = 0; i < steps; i++) {
        let t = (f32(i) + 0.5) / f32(steps) * span;
        let height = length(origin + direction * t) - PLANET;
        depth += medium(height).extinction * (span / f32(steps));
    }
    return vec4<f32>(exp(-depth), 1.0);
}

// Light scattered more than once, as Hillaire works it out: what reaches a
// point after scattering twice, from all around, and how much of the light
// arriving at a point is scattered on again, which gives the sum of every
// order after.
@fragment
fn multiple_table(in: Fullscreen) -> @location(0) vec4<f32> {
    let sun_mu = in.uv.x * 2.0 - 1.0;
    let r = PLANET + in.uv.y * (TOP - PLANET);
    let origin = vec3<f32>(0.0, r, 0.0);
    let sun = vec3<f32>(sqrt(max(1.0 - sun_mu * sun_mu, 0.0)), sun_mu, 0.0);
    let rings = 8;
    var second = vec3<f32>(0.0);
    var transfer = vec3<f32>(0.0);
    for (var a = 0; a < rings; a++) {
        for (var b = 0; b < rings; b++) {
            // Evenly over the sphere.
            let cos_theta = 1.0 - 2.0 * (f32(a) + 0.5) / f32(rings);
            let phi = 2.0 * PI * (f32(b) + 0.5) / f32(rings);
            let sin_theta = sqrt(max(1.0 - cos_theta * cos_theta, 0.0));
            let direction = vec3<f32>(sin_theta * cos(phi), cos_theta, sin_theta * sin(phi));
            let ground = ray_sphere(origin, direction, PLANET);
            var span = ray_sphere(origin, direction, TOP);
            if ground > 0.0 {
                span = ground;
            }
            span = max(span, 0.0);
            let steps = 20;
            let step = span / f32(steps);
            var through = vec3<f32>(1.0);
            var light = vec3<f32>(0.0);
            var scattered = vec3<f32>(0.0);
            for (var i = 0; i < steps; i++) {
                let point = origin + direction * ((f32(i) + 0.5) * step);
                let height = length(point) - PLANET;
                let air = medium(height);
                let up = normalize(point);
                let reaching = sun_transmittance(length(point), dot(up, sun));
                let scattering = air.rayleigh + vec3<f32>(air.mie);
                let fade = exp(-air.extinction * step);
                let absorbed = (vec3<f32>(1.0) - fade) / max(air.extinction, vec3<f32>(1e-7));
                // An even phase function: 1 / 4π.
                light += through * scattering * reaching * absorbed / (4.0 * PI);
                scattered += through * scattering * absorbed / (4.0 * PI);
                through *= fade;
            }
            if ground > 0.0 {
                let point = origin + direction * ground;
                let up = normalize(point);
                let reaching = sun_transmittance(length(point), dot(up, sun));
                light += through * reaching * max(dot(up, sun), 0.0) * GROUND_ALBEDO / PI;
            }
            second += light;
            transfer += scattered;
        }
    }
    // Gathered from all around with the even phase function: each of the
    // directions stands for 4π / 64 of the sphere, times 1 / 4π.
    let count = f32(rings * rings);
    second = second / count;
    transfer = transfer / count;
    return vec4<f32>(second / (vec3<f32>(1.0) - min(transfer, vec3<f32>(0.99))), 1.0);
}

// The sky seen from the camera, for one direction: sunlight and moonlight
// scattered towards the camera along the way out of the air.
@fragment
fn sky_table(in: Fullscreen) -> @location(0) vec4<f32> {
    let r = radius_at(tables.sun.w);
    let origin = vec3<f32>(0.0, r, 0.0);
    // Back from the table's coordinates to a direction, with the sun's side
    // along x.
    let horizon = sqrt(max(r * r - PLANET * PLANET, 0.0));
    let beta = acos(clamp(horizon / r, -1.0, 1.0));
    let zenith_horizon = PI - beta;
    var angle: f32;
    if in.uv.y < 0.5 {
        let c = 1.0 - in.uv.y * 2.0;
        angle = (1.0 - c * c) * zenith_horizon;
    } else {
        let c = in.uv.y * 2.0 - 1.0;
        angle = zenith_horizon + c * c * beta;
    }
    let view_mu = cos(angle);
    let side = -(in.uv.x * in.uv.x * 2.0 - 1.0);
    let sin_view = sqrt(max(1.0 - view_mu * view_mu, 0.0));
    let direction = vec3<f32>(
        side * sin_view,
        view_mu,
        sqrt(max(1.0 - side * side, 0.0)) * sin_view,
    );
    let sun_mu = tables.sun.y;
    let sun = vec3<f32>(sqrt(max(1.0 - sun_mu * sun_mu, 0.0)), sun_mu, 0.0);
    let moon = -sun;

    let ground = ray_sphere(origin, direction, PLANET);
    var span = ray_sphere(origin, direction, TOP);
    if ground > 0.0 {
        span = ground;
    }
    span = max(span, 0.0);
    let steps = 32;
    var through = vec3<f32>(1.0);
    var color = vec3<f32>(0.0);
    var travelled = 0.0;
    for (var i = 0; i < steps; i++) {
        // Steps grow with distance: the air near the camera matters most.
        let next = span * pow((f32(i) + 1.0) / f32(steps), 2.0);
        let step = next - travelled;
        let point = origin + direction * (travelled + step * 0.5);
        travelled = next;
        let height = length(point) - PLANET;
        let air = medium(height);
        let up = normalize(point);
        var gathered = vec3<f32>(0.0);
        for (var body = 0; body < 2; body++) {
            let toward = select(sun, moon, body == 1);
            let strength = select(SUN_LIGHT, SUN_LIGHT * MOON_LIGHT * MOON_TINT, body == 1);
            let cos_angle = dot(direction, toward);
            let mu = dot(up, toward);
            let reaching = sun_transmittance(length(point), mu);
            let single = air.rayleigh * rayleigh_phase(cos_angle) + vec3<f32>(air.mie * mie_phase(cos_angle));
            let more = textureSampleLevel(multiple, table_sampler, multiple_uv(length(point), mu), 0.0).rgb
                * (air.rayleigh + vec3<f32>(air.mie));
            gathered += strength * (reaching * single + more);
        }
        let fade = exp(-air.extinction * step);
        color += through * gathered * (vec3<f32>(1.0) - fade) / max(air.extinction, vec3<f32>(1e-7));
        through *= fade;
    }
    return vec4<f32>(color, 1.0);
}

// Two texels of light at the camera. The first: what the whole sky sheds
// on a surface facing up, the sky table summed over the half above, each
// direction weighted by how steeply it falls. The second: the light
// straight from the sun, or by night from the moon.
@fragment
fn ambient_table(in: Fullscreen) -> @location(0) vec4<f32> {
    let r = radius_at(tables.sun.w);
    if in.clip.x > 1.0 {
        let sun_mu = tables.sun.y;
        let sun = SUN_LIGHT * sun_transmittance(r, sun_mu);
        let moon = SUN_LIGHT * MOON_LIGHT * MOON_TINT * sun_transmittance(r, -sun_mu);
        return vec4<f32>(select(moon, sun, sun_mu >= 0.0), 1.0);
    }
    var total = vec3<f32>(0.0);
    let rings = 8;
    for (var a = 0; a < rings; a++) {
        for (var b = 0; b < rings * 2; b++) {
            // Even in the cosine of the angle from straight up, so each
            // sample stands for the same share of the sky's light.
            let mu = sqrt((f32(a) + 0.5) / f32(rings));
            let side = cos(PI * (f32(b) + 0.5) / f32(rings * 2) * 2.0);
            total += textureSampleLevel(sky_view, table_sampler, sky_uv(r, mu, side), 0.0).rgb;
        }
    }
    // Irradiance is π times the average when samples follow the cosine.
    return vec4<f32>(total * PI / f32(rings * rings * 2), 1.0);
}
