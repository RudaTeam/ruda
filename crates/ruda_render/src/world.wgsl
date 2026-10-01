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
    // Light space of each shadow cascade, near then far.
    shadow_view_proj: array<mat4x4<f32>, 2>,
    // xyz: towards the sun or moon, whichever casts shadows; w: how strong
    // its direct light is, 0 without shadows.
    shadow_light: vec4<f32>,
    // x: distance where the far cascade takes over; y: where shadows end;
    // z: unused; w: size of a shadow map texel in texture coordinates.
    shadow: vec4<f32>,
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
// Depth from the light, a layer per cascade.
@group(0) @binding(5) var shadow_map: texture_depth_2d_array;
@group(0) @binding(6) var shadow_sampler: sampler_comparison;
// The light space of the cascade being drawn into the shadow map.
@group(1) @binding(0) var<uniform> shadow_pass: mat4x4<f32>;

// How bright a light level from 0 to 1 looks: steep near full light, like
// light falling off around a torch.
fn brightness(level: vec4<f32>) -> vec4<f32> {
    return level / (4.0 - 3.0 * level);
}

struct ChunkVertex {
    @builtin(position) clip: vec4<f32>,
    // Relative to the camera.
    @location(7) position: vec3<f32>,
    @location(8) @interpolate(flat) normal: vec3<f32>,
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

// One corner of a packed quad.
struct Corner {
    // Relative to the chunk.
    local: vec3<f32>,
    // Which corner: u + 2v.
    index: u32,
    uv: vec2<f32>,
    face: u32,
}

fn quad_corner(quad: vec4<u32>, vertex: u32) -> Corner {
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

    var out: Corner;
    // Each quad reaches a hair past its edges: where a corner of one quad
    // lies on the edge of another, rounding could otherwise leave pixel-wide
    // cracks.
    let overlap = 0.001;
    out.local = block
        + unit(axis) * select(0.0, 1.0, positive)
        + unit(u_axis) * (f32(cu) * width + (f32(cu) * 2.0 - 1.0) * overlap)
        + unit(v_axis) * (f32(cv) * height + (f32(cv) * 2.0 - 1.0) * overlap);
    out.index = cu + 2u * cv;
    // Image rows run downwards, so side faces flip v to keep textures upright.
    // Texture coordinates beyond 1 repeat the texture across merged faces.
    out.uv = select(
        vec2<f32>(f32(cu) * width, f32(cv) * height),
        vec2<f32>(f32(cu) * width, (1.0 - f32(cv)) * height),
        axis != 1u,
    );
    out.face = face;
    return out;
}

// A position in the chunk in `slot`, relative to the camera.
fn camera_relative(slot: u32, local: vec3<f32>) -> vec3<f32> {
    let texel = vec2<i32>(i32(slot % ORIGINS_WIDTH), i32(slot / ORIGINS_WIDTH));
    let origin = textureLoad(chunk_origins, texel, 0).xyz;
    return vec3<f32>(origin - globals.camera_block.xyz) + local - globals.camera_fract.xyz;
}

fn face_normal(face: u32) -> vec3<f32> {
    return unit(face / 2u) * select(-1.0, 1.0, face % 2u == 0u);
}

fn unpack_light(bits: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(bits & 15u),
        f32((bits >> 4u) & 15u),
        f32((bits >> 8u) & 15u),
        f32((bits >> 12u) & 15u),
    ) / 15.0;
}

// Fixed shading per side, so the shape of the terrain reads.
fn face_shade(face: u32) -> f32 {
    var shades = array<f32, 6>(0.8, 0.8, 1.0, 0.55, 0.7, 0.7);
    return shades[face];
}

@vertex
fn chunk_vertex(@builtin(vertex_index) vertex: u32, @location(0) quad: vec4<u32>) -> ChunkVertex {
    let corner = quad_corner(quad, vertex);
    let position = camera_relative(quad.y >> 18u, corner.local);

    var out: ChunkVertex;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.position = position;
    out.normal = face_normal(corner.face);
    out.uv = corner.uv;
    out.layer = quad.y & 0x3ffu;
    out.shade = face_shade(corner.face);
    out.distance = length(position);
    let lights = select(quad.z, quad.w, corner.index >= 2u) >> (16u * (corner.index & 1u));
    out.light = unpack_light(lights);
    out.occlusion = f32((quad.y >> (10u + 2u * corner.index)) & 3u) / 3.0;
    out.glows = (quad.x >> 29u) & 1u;
    return out;
}

// Chunk quads seen from the sun, into the shadow map.
@vertex
fn shadow_vertex(@builtin(vertex_index) vertex: u32, @location(0) quad: vec4<u32>) -> @builtin(position) vec4<f32> {
    let corner = quad_corner(quad, vertex);
    return shadow_pass * vec4<f32>(camera_relative(quad.y >> 18u, corner.local), 1.0);
}

// Blocks that aren't cubes, as triangles; see `ModelVertex` in `mesh.rs`.
@vertex
fn model_vertex(@location(0) vertex: vec4<u32>) -> ChunkVertex {
    let local = vec3<f32>(
        f32(vertex.x & 1023u),
        f32((vertex.x >> 10u) & 1023u),
        f32((vertex.x >> 20u) & 1023u),
    ) / 16.0 - 16.0;
    let position = camera_relative(vertex.y >> 18u, local);
    let face = (vertex.z >> 16u) & 7u;

    var out: ChunkVertex;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.position = position;
    out.normal = face_normal(face);
    out.uv = vec2<f32>(f32(vertex.y & 31u), f32((vertex.y >> 5u) & 31u)) / 16.0;
    out.layer = (vertex.y >> 10u) & 255u;
    out.shade = face_shade(face);
    out.distance = length(position);
    out.light = unpack_light(vertex.z & 0xffffu);
    out.occlusion = 1.0;
    out.glows = (vertex.x >> 30u) & 1u;
    return out;
}

// How much of the sun's (or moon's) direct light reaches a point: 0 in full
// shadow, 1 in full light.
fn sunlit(position: vec3<f32>, normal: vec3<f32>, distance: f32) -> f32 {
    let cascade = select(1, 0, distance < globals.shadow.x);
    // Nudged out along the normal so faces don't shade themselves.
    let lifted = position + normal * select(0.15, 0.05, cascade == 0);
    let light = globals.shadow_view_proj[cascade] * vec4<f32>(lifted, 1.0);
    let uv = vec2<f32>(light.x * 0.5 + 0.5, 0.5 - light.y * 0.5);
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) || light.z >= 1.0 {
        return 1.0;
    }
    // Four filtered taps around the point soften the edge.
    var lit = 0.0;
    for (var i = 0u; i < 4u; i++) {
        let offset = (vec2<f32>(f32(i & 1u), f32(i >> 1u)) - 0.5) * globals.shadow.w;
        lit += textureSampleCompareLevel(shadow_map, shadow_sampler, uv + offset, cascade, light.z);
    }
    // Shadows fade out where the shadow map ends.
    let fade = smoothstep(globals.shadow.y * 0.85, globals.shadow.y, distance);
    return mix(lit * 0.25, 1.0, fade);
}

@fragment
fn chunk_fragment(in: ChunkVertex) -> @location(0) vec4<f32> {
    let texel = textureSample(block_textures, block_sampler, in.uv, in.layer).rgb;
    let level = brightness(in.light);
    // With shadows, part of sky light comes straight from the sun or moon
    // and the rest from the whole sky; without, a fixed shade per side.
    var direction = in.shade;
    if globals.shadow_light.w > 0.0 {
        let facing = max(dot(in.normal, globals.shadow_light.xyz), 0.0);
        let direct = facing * sunlit(in.position, in.normal, in.distance);
        direction = mix(in.shade, 0.5 * in.shade + 0.65 * direct, globals.shadow_light.w);
    }
    let sky = globals.sky_light.rgb * level.x * direction;
    let light = max(max(sky, level.yzw * in.shade), vec3<f32>(globals.sky_light.w * in.shade));
    var color = texel * light * mix(0.45, 1.0, in.occlusion);
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
