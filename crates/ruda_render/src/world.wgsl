// The sky, the block world and the outline of the targeted block, in
// linear light that may be far brighter than white; `post.wgsl` maps it to
// the screen.
//
// Chunks are drawn as instanced quads: each instance is one packed quad (see
// `mesh.rs`) and the vertex shader works out its four corners. Bits 18 to 31
// of the quad's second word are its chunk's slot (see `world_pass.rs`).

struct Globals {
    view_proj: mat4x4<f32>,
    // Turns screen positions back into directions, for the sky.
    inverse_view_proj: mat4x4<f32>,
    // x: distance where the haze at the edge of the view starts, y: where it
    // hides everything; z: where far terrain gives way to chunks drawn in
    // full.
    fog: vec4<f32>,
    // xyz: minimum corner of the targeted block, relative to the camera.
    selection: vec4<f32>,
    // xy: viewport size in pixels.
    screen: vec4<f32>,
    // The block the camera is in and the camera's position inside it. Chunk
    // positions are integers, so subtracting them stays exact anywhere.
    camera_block: vec4<i32>,
    camera_fract: vec4<f32>,
    // xyz: towards the sun (the moon is opposite); w: 0 at night, 1 by day.
    sun: vec4<f32>,
    // x: how bright the stars are; y: how much of the sky the camera sees,
    // 1 outside, little deep in a cave; z: the least light anything gets;
    // w: how much of the sky clouds cover.
    sky: vec4<f32>,
    // Light space of each shadow cascade, near then far.
    shadow_view_proj: array<mat4x4<f32>, 2>,
    // xyz: towards the sun or moon, whichever casts shadows; w: how strong
    // its shadows are, 0 without shadows.
    shadow_light: vec4<f32>,
    // x: distance where the far cascade takes over; y: where shadows end;
    // z: unused; w: size of a shadow map texel in texture coordinates.
    shadow: vec4<f32>,
    // xyz: the first cell of the clouds near the camera, its bottom corner,
    // relative to the camera; w: edge length of a cell.
    cloud_origin: vec4<f32>,
    // x: how thick clouds are; y: 1 with clouds, 0 without; z, w: where
    // they start and finish fading out with distance.
    cloud: vec4<f32>,
    // x: how far it is from one keyframe of `cloud_density` to the next,
    // 0 to 1; y: the density from which a cell is fully opaque.
    cloud_shape: vec4<f32>,
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
// How dense the clouds are, a cell per texel from the first cell: r at one
// keyframe, g at the next; see `clouds.rs`.
@group(0) @binding(7) var cloud_density: texture_2d<f32>;
@group(0) @binding(8) var smooth_sampler: sampler;
// Width of `cloud_density`, in cells.
const CLOUD_MASK_SIZE = 256.0;

// How opaque a cloud of some density is, and how much it shades.
fn cloud_opacity(density: f32) -> f32 {
    return smoothstep(0.0, globals.cloud_shape.y, density);
}
// The air, see `sky_tables.wgsl`: how much light gets through it, the sky
// seen from the camera, and two texels of light at the camera, from the
// whole sky and straight from the sun or moon.
@group(0) @binding(9) var transmittance: texture_2d<f32>;
@group(0) @binding(10) var sky_view: texture_2d<f32>;
@group(0) @binding(11) var ambient: texture_2d<f32>;
// The light space of the cascade being drawn into the shadow map.
@group(1) @binding(0) var<uniform> shadow_pass: mat4x4<f32>;

// How bright glowing blocks are, enough to bloom.
const GLOW = 6.0;
// How bright the sun's and moon's images are, before the air.
const SUN_DISK = 30.0;
const MOON_DISK = 1.5;
// How bright the stars are on a clear night.
const STARS = 0.05;
// Light from blocks at full level, like a torch right next to a wall.
const BLOCK = vec3<f32>(1.6, 1.6, 1.6);
// How much of the light reaching the ground the ground throws back up,
// warm from grass and earth.
const GROUND = vec3<f32>(0.3, 0.27, 0.2);
// How strong the sky's light is against the sun's.
const SKY = 0.65;
// How thick the haze is per block of air at sea level, and how quickly
// the air thins with height, in blocks: at sea level, a thousand blocks of
// air hide about half of what's behind.
const HAZE = 0.6e-3;
const HAZE_HEIGHT = 120.0;
// The haze hides blue soonest, so distant land turns blue.
const HAZE_COLOR = vec3<f32>(0.6, 0.8, 1.0);
// How much a cloud dims the sun under it.
const CLOUD_SHADOW = 0.75;
const CLOUD_ALBEDO = 0.9;
// How high clouds are lit as if they were, in blocks.
const CLOUD_LIGHT_HEIGHT = 2000.0;

// The camera's height above sea level, in blocks.
fn camera_height() -> f32 {
    return f32(globals.camera_block.y) + globals.camera_fract.y;
}

// The sky in `direction`, seen from the camera, greyer and dimmer under
// clouds.
fn sky_color(direction: vec3<f32>) -> vec3<f32> {
    let r = radius_at(camera_height());
    let uv = sky_uv(r, direction.y, sun_side(direction, globals.sun.xyz));
    let color = textureSampleLevel(sky_view, smooth_sampler, uv, 0.0).rgb;
    let cover = globals.sky.w;
    let grey = dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(color, vec3<f32>(grey), cover * 0.5) * (1.0 - 0.2 * cover);
}

// How much light from above the air gets through it to the camera from a
// direction `mu` from straight up; none from below the horizon.
fn air_transmittance(mu: f32) -> vec3<f32> {
    return transmittance_from(radius_at(camera_height()), mu);
}

fn transmittance_from(r: f32, mu: f32) -> vec3<f32> {
    if below_horizon(r, mu) {
        return vec3<f32>(0.0);
    }
    return textureSampleLevel(transmittance, smooth_sampler, transmittance_uv(r, mu), 0.0).rgb;
}

// What the whole sky sheds on a surface facing up. Eyes take sunlight as
// white and only partly see shade as blue: the sky's light loses some of
// its colour.
fn skylight() -> vec3<f32> {
    let light = textureLoad(ambient, vec2<i32>(0, 0), 0).rgb;
    let grey = dot(light, vec3<f32>(0.2126, 0.7152, 0.0722));
    return mix(light, vec3<f32>(grey), 0.5) * SKY * (1.0 - 0.15 * globals.sky.w);
}

// What the sun, or by night the moon, sheds on a surface facing it.
fn direct_light() -> vec3<f32> {
    return textureLoad(ambient, vec2<i32>(1, 0), 0).rgb * (1.0 - 0.5 * globals.sky.w);
}

// The air between the camera and a point: the farther, the more the point
// takes on the colour of the sky behind it, blue at noon, warm towards a
// low sun; thickest low down. At the edge of the view, the world fades
// into the sky.
fn through_air(color: vec3<f32>, position: vec3<f32>, distance: f32) -> vec3<f32> {
    let direction = position / max(distance, 1e-4);
    let along = normalize(vec3<f32>(direction.x, max(abs(direction.y) * 0.3, 0.02), direction.z));
    let behind = sky_color(along) * globals.sky.y;
    // The air's thickness falls off with height: its average along the way.
    let low = exp(-max(camera_height(), 0.0) / HAZE_HEIGHT);
    let high = exp(-max(camera_height() + position.y, 0.0) / HAZE_HEIGHT);
    var thickness = low;
    if abs(low - high) > 1e-4 {
        thickness = (low - high) / log(low / high);
    }
    let fade = exp(-HAZE_COLOR * (HAZE * thickness * distance));
    let hazed = color * fade + behind * (vec3<f32>(1.0) - fade);
    return mix(hazed, behind, smoothstep(globals.fog.x, globals.fog.y, distance));
}

// How bright a light level from 0 to 1 looks: steep near full light, like
// light falling off around a torch.
fn brightness(level: vec4<f32>) -> vec4<f32> {
    return level / (4.0 - 3.0 * level);
}

struct ChunkVertex {
    @builtin(position) clip: vec4<f32>,
    // Relative to the camera.
    @location(3) position: vec3<f32>,
    @location(7) @interpolate(flat) normal: vec3<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) layer: u32,
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
    return shade(in);
}

// A corner of the `face` of the box from `start` of `size`, counter-clockwise
// from outside, and the axes the face spans.
struct BoxCorner {
    local: vec3<f32>,
    u_axis: u32,
    v_axis: u32,
}

fn box_corner(start: vec3<f32>, size: vec3<f32>, face: u32, vertex: u32) -> BoxCorner {
    let axis = face / 2u;
    var out: BoxCorner;
    out.u_axis = 0u;
    out.v_axis = 2u;
    if axis == 0u {
        out.u_axis = 2u;
        out.v_axis = 1u;
    } else if axis == 2u {
        out.v_axis = 1u;
    }
    var cu = vertex & 1u;
    var cv = vertex >> 1u;
    if face == 0u || face == 2u || face == 5u {
        let swap = cu;
        cu = cv;
        cv = swap;
    }
    out.local = start
        + unit(axis) * select(0.0, size[axis], face % 2u == 0u)
        + unit(out.u_axis) * (f32(cu) * size[out.u_axis])
        + unit(out.v_axis) * (f32(cv) * size[out.v_axis]);
    return out;
}

struct CloudVertex {
    // The same in the depth and the colour pass, so the depth test between
    // them is exact.
    @builtin(position) @invariant clip: vec4<f32>,
    // Relative to the camera. Distances are taken per pixel: a cloud can be
    // hundreds of blocks across, and its corners say little about its middle.
    @location(0) position: vec3<f32>,
    @location(1) @interpolate(flat) normal: vec3<f32>,
    // 0 at the bottom of the cloud, 1 at the top.
    @location(2) height: f32,
}

// Cloud boxes; see `CloudMesh` in `clouds.rs`.
@vertex
fn cloud_vertex(@builtin(vertex_index) vertex: u32, @location(0) quad: vec4<u32>) -> CloudVertex {
    let cell = globals.cloud_origin.w;
    let start = vec3<f32>(f32(quad.x & 1023u), 0.0, f32((quad.x >> 10u) & 1023u)) * cell;
    let size = vec3<f32>(
        f32((quad.y & 1023u) + 1u) * cell,
        globals.cloud.x,
        f32(((quad.y >> 10u) & 1023u) + 1u) * cell,
    );
    let face = (quad.x >> 20u) & 7u;
    let local = box_corner(start, size, face, vertex).local;
    let position = globals.cloud_origin.xyz + local;

    var out: CloudVertex;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.position = position;
    out.normal = face_normal(face);
    out.height = local.y / globals.cloud.x;
    return out;
}

@fragment
fn cloud_fragment(in: CloudVertex) -> @location(0) vec4<f32> {
    // The cell this face belongs to: a little inside it, off the edge it
    // shares with the next cell.
    let inside = in.position - in.normal * 0.5;
    let cell = vec2<i32>(floor((inside.xz - globals.cloud_origin.xz) / globals.cloud_origin.w));
    let keyframes = textureLoad(cloud_density, clamp(cell, vec2<i32>(0), vec2<i32>(255)), 0).rg;
    let density = mix(keyframes.x, keyframes.y, globals.cloud_shape.x);
    let opacity = cloud_opacity(density);

    // Lit by the sun or moon where a face turns to it, by the sky all
    // over: most on tops, less on walls, least underneath, the less the
    // denser the cloud. Undersides catch a low sun, which shines under the
    // clouds at sunrise and sunset.
    // The sun and moon light clouds as if they were as high as real ones,
    // so they catch the sun before it rises and after it sets.
    let sun = globals.sun.xyz;
    let r = radius_at(CLOUD_LIGHT_HEIGHT);
    let sunlight = SUN_LIGHT * transmittance_from(r, sun.y);
    let moonlight = SUN_LIGHT * MOON_LIGHT * MOON_TINT * transmittance_from(r, -sun.y);
    let dense = min(density * 2.0, 1.0);
    let low_sun = 1.0 - smoothstep(0.0, 0.3, abs(sun.y));
    var lit = vec2<f32>(max(dot(in.normal, sun), 0.0), max(dot(in.normal, -sun), 0.0));
    var around = 1.0;
    if in.normal.y < -0.5 {
        lit = vec2<f32>(0.6 * low_sun, 0.0);
        around = 0.45 * (1.0 - 0.35 * dense);
    } else if in.normal.y < 0.5 {
        around = 0.65 * mix(0.85, 1.0, in.height);
    }
    let light = (sunlight * lit.x + moonlight * lit.y) * (1.0 - 0.5 * globals.sky.w);
    var color = CLOUD_ALBEDO * (light + skylight() * around) / PI;
    // Thin clouds between the camera and the sun glow.
    let view = normalize(in.position);
    let behind = pow(max(dot(view, sun), 0.0), 8.0);
    color += sunlight * behind * (1.0 - opacity * 0.6) * 0.05;
    color = through_air(color, in.position, length(in.position));
    let fade = 1.0 - smoothstep(globals.cloud.z, globals.cloud.w, length(in.position.xz));
    return vec4<f32>(color, 0.92 * opacity * fade);
}

// Far-away terrain; see `LodQuad` in `lod.rs`. A tile's position comes from
// the chunk position texture, like a chunk's.
@vertex
fn lod_vertex(@builtin(vertex_index) vertex: u32, @location(0) quad: vec4<u32>) -> ChunkVertex {
    let first = vec3<f32>(f32(quad.x & 63u), 0.0, f32((quad.x >> 6u) & 63u)) * 4.0;
    let cells = vec3<f32>(f32(((quad.x >> 12u) & 63u) + 1u), 0.0, f32(((quad.x >> 18u) & 63u) + 1u));
    let face = (quad.x >> 24u) & 7u;
    // The box's bottom and top, sign-extended from 16 bits.
    let low = f32(bitcast<i32>(quad.z << 16u) >> 16u);
    let high = f32(bitcast<i32>(quad.z) >> 16u);
    let start = vec3<f32>(first.x, low, first.z);
    let size = vec3<f32>(cells.x * 4.0, high - low, cells.z * 4.0);
    let corner = box_corner(start, size, face, vertex);
    var local = corner.local;
    // A little below the chunks drawn in full, so where both are drawn,
    // those win.
    local.y -= 0.5;
    let position = camera_relative(quad.y >> 18u, local);

    var out: ChunkVertex;
    out.clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.position = position;
    out.normal = face_normal(face);
    // One texture per block, upright on walls.
    out.uv = select(
        vec2<f32>(local[corner.u_axis], -local[corner.v_axis]),
        local.xz,
        face / 2u == 1u,
    );
    out.layer = quad.y & 0x3ffu;
    out.light = vec4<f32>(1.0, 0.0, 0.0, 0.0);
    out.occlusion = 1.0;
    out.glows = 0u;
    return out;
}

@fragment
fn lod_fragment(in: ChunkVertex) -> @location(0) vec4<f32> {
    // Near the camera, the chunks drawn in full take over.
    if length(in.position.xz) < globals.fog.z {
        discard;
    }
    return shade(in);
}

fn shade(in: ChunkVertex) -> vec4<f32> {
    // Per pixel: far-away tops are up to a tile across, too big to take
    // their corners' distances.
    let distance = length(in.position);
    let albedo = textureSample(block_textures, block_sampler, in.uv, in.layer).rgb;
    let level = brightness(in.light);
    if in.glows == 1u {
        return vec4<f32>(through_air(albedo * GLOW, in.position, distance), 1.0);
    }
    let occlusion = mix(0.45, 1.0, in.occlusion);
    // Straight from the sun or moon, on faces turned to it, unless
    // something stands in between: the shadow map where there is one,
    // and where there isn't, how open the sky above is.
    let toward = globals.shadow_light.xyz;
    let facing = max(dot(in.normal, toward), 0.0);
    var seen = smoothstep(0.55, 1.0, level.x);
    if globals.shadow_light.w > 0.0 {
        let shadowed = mix(1.0, sunlit(in.position, in.normal, distance), globals.shadow_light.w);
        seen = shadowed * smoothstep(0.0, 0.4, level.x);
    }
    seen *= 1.0 - CLOUD_SHADOW * cloud_cover(in.position, toward);
    let sun = direct_light();
    let direct = sun * facing * seen;
    // From the whole sky: most on tops, less on walls, and on walls and
    // undersides some thrown back by the ground. How much of the sky
    // reaches a point is its sky light.
    let sky = skylight();
    let up = in.normal.y;
    let around = sky * (0.55 + 0.45 * up)
        + (sun * max(toward.y, 0.0) + sky) * GROUND * (0.5 - 0.5 * up);
    let ambient = around * level.x * occlusion;
    let blocks = BLOCK * level.yzw * occlusion;
    let least = vec3<f32>(globals.sky.z) * occlusion;
    let color = albedo * ((direct + ambient) / PI + blocks + least);
    return vec4<f32>(through_air(color, in.position, distance), 1.0);
}

// How much the clouds between a point and the sun or moon `toward` it
// cover it, 0 to about 1.
fn cloud_cover(position: vec3<f32>, toward: vec3<f32>) -> f32 {
    let height = globals.cloud_origin.y + globals.cloud.x * 0.5 - position.y;
    if globals.cloud.y <= 0.0 || toward.y <= 0.02 || height <= 0.0 {
        return 0.0;
    }
    let through = position + toward * (height / toward.y);
    let cell = (through.xz - globals.cloud_origin.xz) / globals.cloud_origin.w;
    let keyframes = textureSampleLevel(cloud_density, smooth_sampler, cell / CLOUD_MASK_SIZE, 0.0).rg;
    let density = mix(keyframes.x, keyframes.y, globals.cloud_shape.x);
    // Denser clouds cast darker shadows.
    return cloud_opacity(density) * mix(0.8, 1.0, min(density * 2.0, 1.0))
        * smoothstep(0.02, 0.2, toward.y);
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
    // Below the horizon, past the edge of the world, the haze goes on.
    var color = sky_color(normalize(vec3<f32>(direction.x, max(direction.y, 0.02), direction.z)));

    // Stars: one in a few hundred cells of a grid around the camera.
    let up = direction.y;
    if globals.sky.x > 0.0 && up > 0.0 {
        let scale = 150.0;
        let cell = floor(direction * scale);
        let chance = hash(cell);
        if chance > 0.995 {
            let jitter = vec3<f32>(hash(cell + 1.3), hash(cell + 2.7), hash(cell + 5.1)) - 0.5;
            let star = normalize((cell + 0.5 + jitter * 0.5) / scale);
            let spread = length(direction - star) * scale;
            let twinkle = (chance - 0.995) / 0.005;
            color += vec3<f32>(smoothstep(0.35, 0.0, spread) * (0.4 + 0.6 * twinkle))
                * globals.sky.x
                * STARS
                * clamp(up * 5.0, 0.0, 1.0)
                * (1.0 - globals.sky.w);
        }
    }
    return vec4<f32>(color * globals.sky.y, 1.0);
}

struct CelestialVertex {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
    @location(1) @interpolate(flat) body: u32,
    @location(2) alpha: f32,
    @location(3) light: vec3<f32>,
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
        * globals.sky.y;
    // Seen through the air: the sun reddens near the horizon.
    out.light = air_transmittance(max(direction.y, 0.0)) * select(SUN_DISK, MOON_DISK, body == 1u)
        * (1.0 - 0.7 * globals.sky.w);
    return out;
}

@fragment
fn celestial_fragment(in: CelestialVertex) -> @location(0) vec4<f32> {
    let texel = textureSample(sky_textures, block_sampler, in.uv, in.body);
    return vec4<f32>(texel.rgb * in.light, texel.a * in.alpha);
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
