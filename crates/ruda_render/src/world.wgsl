// The sky, the block world, the outline of the targeted block and the
// crosshair.
//
// Chunks are drawn as instanced quads: each instance is one packed quad (see
// `mesh.rs`) and the vertex shader works out its four corners. Bits 18 to 31
// of the quad's second word are its chunk's slot (see `world_pass.rs`).

struct Globals {
    view_proj: mat4x4<f32>,
    // Turns screen positions back into directions, for the sky.
    inverse_view_proj: mat4x4<f32>,
    fog_color: vec4<f32>,
    // x: distance where fog starts, y: distance where it hides everything.
    fog: vec4<f32>,
    // xyz: minimum corner of the targeted block, relative to the camera.
    selection: vec4<f32>,
    // xy: viewport size in pixels.
    screen: vec4<f32>,
    // The block the camera is in and the camera's position inside it. Chunk
    // positions are integers, so subtracting them stays exact anywhere.
    camera_block: vec4<i32>,
    camera_fract: vec4<f32>,
    // rgb: colour and strength of full sky light; w: the least light
    // anything gets.
    sky_light: vec4<f32>,
    // xyz: towards the sun (the moon is opposite); w: 0 at night, 1 by day.
    sun: vec4<f32>,
    // rgb: sky colour straight up; w: how bright the stars are.
    zenith: vec4<f32>,
    // rgb: sky colour at the horizon.
    horizon: vec4<f32>,
    // rgb: glow around a low sun.
    glow: vec4<f32>,
}

// Width of `chunk_origins`, whose texels are chunk slots.
const ORIGINS_WIDTH = 128u;

@group(0) @binding(0) var<uniform> globals: Globals;
@group(0) @binding(1) var block_textures: texture_2d_array<f32>;
@group(0) @binding(2) var block_sampler: sampler;
// xyz: the world position of the chunk in each slot.
@group(0) @binding(3) var chunk_origins: texture_2d<i32>;
// Layer 0 is the sun, layer 1 the moon.
@group(0) @binding(4) var sky_textures: texture_2d_array<f32>;

// How bright a light level from 0 to 1 looks: steep near full light, like
// light falling off around a torch.
fn brightness(level: vec4<f32>) -> vec4<f32> {
    return level / (4.0 - 3.0 * level);
}

struct ChunkVertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
    @location(2) shade: f32,
    @location(3) distance: f32,
    // Sky, red, green and blue light, each from 0 to 1.
    @location(4) light: vec4<f32>,
    // 0 where the corner is darkest, 1 where nothing shades it.
    @location(5) occlusion: f32,
    @location(6) @interpolate(flat) glows: u32,
}

fn unit(axis: u32) -> vec3<f32> {
    return vec3<f32>(f32(axis == 0u), f32(axis == 1u), f32(axis == 2u));
}

@vertex
fn chunk_vertex(@builtin(vertex_index) vertex: u32, @location(0) quad: vec4<u32>) -> ChunkVertex {
    let block = vec3<f32>(
        f32(quad.x & 31u),
        f32((quad.x >> 5u) & 31u),
        f32((quad.x >> 10u) & 31u),
    );
    let width = f32(((quad.x >> 15u) & 31u) + 1u);
    let height = f32(((quad.x >> 20u) & 31u) + 1u);
    let face = (quad.x >> 25u) & 7u;
    let flip = ((quad.x >> 28u) & 1u) == 1u;
    let axis = face / 2u;
    let positive = face % 2u == 0u;

    // u and v span the face; on side faces v is vertical.
    var u_axis = 0u;
    var v_axis = 2u;
    if axis == 0u {
        u_axis = 2u;
        v_axis = 1u;
    } else if axis == 2u {
        v_axis = 1u;
    }

    // Strip corners (0, 0), (1, 0), (0, 1), (1, 1), or (1, 0), (1, 1),
    // (0, 0), (0, 1) to split along the other diagonal. Mirrored on faces
    // where u × v points inwards, so every face winds counter-clockwise
    // from outside.
    var cu = vertex & 1u;
    var cv = vertex >> 1u;
    if flip {
        cu = select(0u, 1u, vertex < 2u);
        cv = vertex & 1u;
    }
    if face == 0u || face == 2u || face == 5u {
        let swap = cu;
        cu = cv;
        cv = swap;
    }
    let corner = cu + 2u * cv;

    // Each quad reaches a hair past its edges: where a corner of one quad
    // lies on the edge of another, rounding could otherwise leave pixel-wide
    // cracks.
    let overlap = 0.001;
    let local = block
        + unit(axis) * select(0.0, 1.0, positive)
        + unit(u_axis) * (f32(cu) * width + (f32(cu) * 2.0 - 1.0) * overlap)
        + unit(v_axis) * (f32(cv) * height + (f32(cv) * 2.0 - 1.0) * overlap);
    let slot = quad.y >> 18u;
    let texel = vec2<i32>(i32(slot % ORIGINS_WIDTH), i32(slot / ORIGINS_WIDTH));
    let origin = textureLoad(chunk_origins, texel, 0).xyz;
    let position = vec3<f32>(origin - globals.camera_block.xyz) + local - globals.camera_fract.xyz;

    var out: ChunkVertex;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    // Image rows run downwards, so side faces flip v to keep textures upright.
    // Texture coordinates beyond 1 repeat the texture across merged faces.
    out.uv = select(
        vec2<f32>(f32(cu) * width, f32(cv) * height),
        vec2<f32>(f32(cu) * width, (1.0 - f32(cv)) * height),
        axis != 1u,
    );
    out.layer = quad.y & 0x3ffu;
    // Fixed shading per side, so the shape of the terrain reads.
    var shades = array<f32, 6>(0.8, 0.8, 1.0, 0.55, 0.7, 0.7);
    out.shade = shades[face];
    out.distance = length(position);

    let lights = select(quad.z, quad.w, corner >= 2u) >> (16u * (corner & 1u));
    out.light = vec4<f32>(
        f32(lights & 15u),
        f32((lights >> 4u) & 15u),
        f32((lights >> 8u) & 15u),
        f32((lights >> 12u) & 15u),
    ) / 15.0;
    let occlusion = (quad.y >> (10u + 2u * corner)) & 3u;
    out.occlusion = f32(occlusion) / 3.0;
    out.glows = (quad.x >> 29u) & 1u;
    return out;
}

@fragment
fn chunk_fragment(in: ChunkVertex) -> @location(0) vec4<f32> {
    let texel = textureSample(block_textures, block_sampler, in.uv, in.layer).rgb;
    let level = brightness(in.light);
    let sky = globals.sky_light.rgb * level.x;
    let light = max(max(sky, level.yzw), vec3<f32>(globals.sky_light.w));
    var color = texel * light * mix(0.45, 1.0, in.occlusion) * in.shade;
    if in.glows == 1u {
        color = texel;
    }
    let fog = smoothstep(globals.fog.x, globals.fog.y, in.distance);
    return vec4<f32>(mix(color, globals.fog_color.rgb, fog), 1.0);
}

struct SkyVertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) screen: vec2<f32>,
}

// One triangle covering the screen.
@vertex
fn sky_vertex(@builtin(vertex_index) index: u32) -> SkyVertex {
    let screen = vec2<f32>(f32(index == 1u) * 4.0 - 1.0, f32(index == 2u) * 4.0 - 1.0);
    var out: SkyVertex;
    out.clip = vec4<f32>(screen, 0.5, 1.0);
    out.screen = screen;
    return out;
}

fn hash(p: vec3<f32>) -> f32 {
    return fract(sin(dot(p, vec3<f32>(12.9898, 78.233, 37.719))) * 43758.5453);
}

@fragment
fn sky_fragment(in: SkyVertex) -> @location(0) vec4<f32> {
    let far = globals.inverse_view_proj * vec4<f32>(in.screen, 1.0, 1.0);
    let direction = normalize(far.xyz / far.w);
    let up = direction.y;
    // Sunrise and sunset colour the horizon, most of all towards the sun.
    let flat_direction = normalize(direction.xz + vec2<f32>(1e-5, 0.0));
    let flat_sun = normalize(globals.sun.xz + vec2<f32>(1e-5, 0.0));
    let sunward = max(dot(flat_direction, flat_sun), 0.0);
    let horizon = globals.horizon.rgb + globals.glow.rgb * (0.2 + 0.8 * sunward * sunward);
    var color = mix(horizon, globals.zenith.rgb, smoothstep(-0.05, 0.5, up));
    // Below the horizon, a little darker.
    color *= mix(1.0, 0.7, smoothstep(0.0, 0.3, -up));
    // The glow right around the sun.
    let toward_sun = max(dot(direction, globals.sun.xyz), 0.0);
    color += globals.glow.rgb * pow(toward_sun, 24.0) * 0.5;

    // Stars: one in a few hundred cells of a grid around the camera.
    if globals.zenith.w > 0.0 && up > 0.0 {
        let scale = 150.0;
        let cell = floor(direction * scale);
        let chance = hash(cell);
        if chance > 0.995 {
            let jitter = vec3<f32>(hash(cell + 1.3), hash(cell + 2.7), hash(cell + 5.1)) - 0.5;
            let star = normalize((cell + 0.5 + jitter * 0.5) / scale);
            let spread = length(direction - star) * scale;
            let twinkle = (chance - 0.995) / 0.005;
            color += vec3<f32>(smoothstep(0.35, 0.0, spread) * (0.4 + 0.6 * twinkle))
                * globals.zenith.w
                * clamp(up * 5.0, 0.0, 1.0);
        }
    }
    return vec4<f32>(color, 1.0);
}

struct CelestialVertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) body: u32,
    @location(2) alpha: f32,
}

// Two squares facing the camera: the sun, then the moon opposite it.
@vertex
fn celestial_vertex(@builtin(vertex_index) index: u32) -> CelestialVertex {
    var square = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let body = index / 6u;
    let corner = square[index % 6u];
    let direction = select(globals.sun.xyz, -globals.sun.xyz, body == 1u);
    let right = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), direction));
    let up = cross(direction, right);
    let size = select(0.12, 0.09, body == 1u);
    // Inside the far plane even at the shortest view distance.
    let position = (direction + (right * corner.x + up * corner.y) * size) * 60.0;

    var out: CelestialVertex;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.uv = vec2<f32>(corner.x, -corner.y) * 0.5 + 0.5;
    out.body = body;
    // Fade out below the horizon; the moon is faint by day.
    let height = smoothstep(-0.2, 0.0, direction.y);
    out.alpha = select(height, height * mix(1.0, 0.25, globals.sun.w), body == 1u)
        * globals.horizon.w;
    return out;
}

@fragment
fn celestial_fragment(in: CelestialVertex) -> @location(0) vec4<f32> {
    let texel = textureSample(sky_textures, block_sampler, in.uv, in.body);
    return vec4<f32>(texel.rgb, texel.a * in.alpha);
}

// The 12 edges of a unit cube as 24 line endpoints; bits 0, 1, 2 of each
// entry are its x, y, z.
const CUBE_EDGES = array<u32, 24>(
    0u, 1u, 1u, 5u, 5u, 4u, 4u, 0u,
    2u, 3u, 3u, 7u, 7u, 6u, 6u, 2u,
    0u, 2u, 1u, 3u, 5u, 7u, 4u, 6u,
);

@vertex
fn outline_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    var edges = CUBE_EDGES;
    let bits = edges[index];
    let corner = vec3<f32>(f32(bits & 1u), f32((bits >> 1u) & 1u), f32((bits >> 2u) & 1u));
    // Slightly larger than the block so the lines don't sink into its faces.
    let position = globals.selection.xyz - 0.002 + corner * 1.004;
    return globals.view_proj * vec4<f32>(position, 1.0);
}

@fragment
fn outline_fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(0.04, 0.04, 0.04, 1.0);
}

@vertex
fn crosshair_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    // Two bars of six vertices each, sized in pixels.
    var square = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0), vec2<f32>(1.0, -1.0), vec2<f32>(-1.0, 1.0),
        vec2<f32>(-1.0, 1.0), vec2<f32>(1.0, -1.0), vec2<f32>(1.0, 1.0),
    );
    let half_size = select(vec2<f32>(1.0, 9.0), vec2<f32>(9.0, 1.0), index < 6u);
    let pixels = square[index % 6u] * half_size;
    return vec4<f32>(pixels * 2.0 / globals.screen.xy, 0.0, 1.0);
}

@fragment
fn crosshair_fragment() -> @location(0) vec4<f32> {
    return vec4<f32>(0.95, 0.95, 0.95, 1.0);
}
