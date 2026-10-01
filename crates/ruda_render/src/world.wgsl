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
    // x: distance where fog starts hiding the edge of the world, y: where it
    // hides everything, z: where far terrain gives way to chunks, w: haze per
    // block at sea level.
    fog: vec4<f32>,
    // xyz: minimum corner of the targeted block, relative to the camera.
    selection: vec4<f32>,
    // xy: viewport size in pixels; z: exposure, what light is multiplied by
    // before it becomes a colour; w: 1 if the target wants sRGB-encoded
    // colours from the shader.
    screen: vec4<f32>,
    // The block the camera is in and the camera's position inside it. Chunk
    // positions are integers, so subtracting them stays exact anywhere.
    camera_block: vec4<i32>,
    camera_fract: vec4<f32>,
    // xyz: towards the sun (the moon is opposite); w: 0 at night, 1 by day.
    sun: vec4<f32>,
    // xyz: towards whichever of the sun and the moon lights the world; w: how
    // strong its shadows are, 0 without shadows.
    light: vec4<f32>,
    // rgb: its light on a white surface facing it; w: the least light
    // anything gets.
    light_color: vec4<f32>,
    // Sky light on a surface by its normal, divided by pi: constant, x, y and
    // z terms.
    ambient: array<vec4<f32>, 4>,
    // x, y: layers of `sky_table` for the sun and the moon, below 0 for none;
    // z: how bright the moon is next to the sun; w: how bright the stars are.
    sky: vec4<f32>,
    // x: how much clouds cover the sky; y: 1 outside, darker seen from a
    // cave; z: how visible the sun and moon are; w: 1 for light as in
    // classic block games, 0 for light through the air.
    weather: vec4<f32>,
    // rgb: the sun's colour through the air, for its disc.
    sun_disc: vec4<f32>,
    // Light space of each shadow cascade, nearest first.
    shadow_view_proj: array<mat4x4<f32>, 3>,
    // xyz: how far each cascade reaches; w: size of a shadow map texel in
    // texture coordinates.
    shadow: vec4<f32>,
    // x: how much of the sky clouds cover; y: 1 with clouds, 0 without; z,
    // w: the bottom and the top of the cloud layer.
    clouds: vec4<f32>,
    // xy: where the camera is in `cloud_patches`; zw: where it is in
    // `cloud_detail` along x and z. Both move with the wind.
    cloud_offset: vec4<f32>,
    // x: where the camera is in `cloud_detail` along y; yz: the first
    // corner of `cloud_obstacles`, x and z relative to the camera.
    cloud_place: vec4<f32>,
    // xyz: towards the light the clouds get: high up, they still see the
    // sun for a while after it has set on the ground.
    cloud_light: vec4<f32>,
    // rgb: that light on a white surface facing it.
    cloud_light_color: vec4<f32>,
    // xy: the first corner of `cloud_shadows`, x and z relative to the
    // camera; z: how many blocks it spans; w: 1 if it holds the shadows of
    // blocky clouds, 0 for soft ones.
    cloud_shadow: vec4<f32>,
    // The view of the last frame, relative to where the camera was then.
    cloud_past_view_proj: mat4x4<f32>,
    // xyz: add to a point of a cloud (relative to the camera) to get where
    // that bit of cloud was last frame, relative to the camera then; w: which
    // pixel of each 2×2 square of the marched clouds is marched this frame,
    // below 0 to march them all.
    cloud_past: vec4<f32>,
    // xyz: the layer of `shadow_map` each cascade is shown from.
    shadow_layers: vec4<f32>,
    // Classic light (see `ClassicSky` in `sky.rs`): sky colour straight up
    // (w: how bright the stars are), at the horizon, the glow of sunrise and
    // sunset, the fog, sky light (w: the least light anything gets) and a
    // cloud's lit top.
    classic: array<vec4<f32>, 6>,
    // Blocky clouds: xy: where the camera is over the patches of cloud, in
    // blocks, repeating every few times `PATCH_PERIOD`; z: the bottom of
    // their layer; w: how thick it is.
    blocky: vec4<f32>,
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
// Patches of cloud, repeating; see `clouds.rs`.
@group(0) @binding(7) var cloud_patches: texture_2d<f32>;
// Linear filtering, clamped at the edges.
@group(0) @binding(8) var smooth_sampler: sampler;
// Linear filtering, repeating.
@group(0) @binding(10) var wrap_sampler: sampler;
// Puffs for the edges of clouds, repeating in all three directions.
@group(0) @binding(11) var cloud_detail: texture_3d<f32>;
// How high obstacles reach into the clouds around the camera, widened and
// softened, above `OBSTACLE_BASE`.
@group(0) @binding(12) var cloud_obstacles: texture_2d<f32>;
// How much light gets through the clouds for each place where it enters
// their bottom on its way down; see `cloud_shadow_fragment`.
@group(0) @binding(13) var cloud_shadows: texture_2d<f32>;
// Blocky clouds come in cells of this many blocks a side.
const BLOCKY_CELL = 12.0;
// The sky's colour by direction, a layer per height of the light; see
// `atmosphere.rs`.
@group(0) @binding(9) var sky_table: texture_3d<f32>;
// Its width (directions around from the light) and height (above the
// horizon).
const SKY_AZIMUTHS = 32.0;
const SKY_ELEVATIONS = 32.0;
// The light space of the cascade being drawn into the shadow map.
@group(1) @binding(0) var<uniform> shadow_pass: mat4x4<f32>;

const PI = 3.14159265;
const LUMINANCE = vec3<f32>(0.2126, 0.7152, 0.0722);
// How bright full block light is next to the noon sun's 1; the same as
// `BLOCK_LIGHT` in `sky.rs`.
const BLOCK_LIGHT = 0.45;
// How bright glowing blocks are.
const GLOW = 1.4;
// How much light clouds reflect.
const CLOUD_ALBEDO = 0.9;
// Haze thins out with height: by e every this many blocks.
const HAZE_HEIGHT = 100.0;
// The look on top of the filmic curve: contrast and saturation.
const LOOK_POWER = 1.15;
const LOOK_SATURATION = 1.25;

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
// shadow, 1 in full light. Each cascade gives way to the next over the last
// fifth of its reach, and the last one fades out.
fn sunlit(position: vec3<f32>, normal: vec3<f32>, distance: f32, facing: f32) -> f32 {
    let reach = globals.shadow.xyz;
    if distance >= reach.z {
        return 1.0;
    }
    var cascade = 2;
    if distance < reach.y {
        cascade = 1;
    }
    if distance < reach.x {
        cascade = 0;
    }
    var lit = cascade_shadow(cascade, position, normal, facing);
    let blend = smoothstep(reach[cascade] * 0.8, reach[cascade], distance);
    if blend > 0.0 {
        var next = 1.0;
        if cascade < 2 {
            next = cascade_shadow(cascade + 1, position, normal, facing);
        }
        lit = mix(lit, next, blend);
    }
    return lit;
}

fn cascade_shadow(cascade: i32, position: vec3<f32>, normal: vec3<f32>, facing: f32) -> f32 {
    // Nudged out along the normal by about a texel, more where the light
    // grazes the face, so faces don't shade themselves.
    let texel = 2.0 * globals.shadow[cascade] * globals.shadow.w;
    let lifted = position + normal * texel * (1.0 + 2.0 * (1.0 - facing));
    let light = globals.shadow_view_proj[cascade] * vec4<f32>(lifted, 1.0);
    let uv = vec2<f32>(light.x * 0.5 + 0.5, 0.5 - light.y * 0.5);
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) || light.z >= 1.0 {
        return 1.0;
    }
    return tent(uv, light.z, i32(globals.shadow_layers[cascade]));
}

// A tent filter over 5×5 texels from nine filtered comparisons (after
// Castaño, "Shadow Mapping Summary"): the edge is soft and slides smoothly
// as the light moves instead of stepping texel by texel.
fn tent(uv: vec2<f32>, depth: f32, layer: i32) -> f32 {
    let texel = globals.shadow.w;
    let at = uv / texel;
    let base = floor(at + 0.5);
    let st = at + 0.5 - base;
    let base_uv = (base - 0.5) * texel;
    let uw = vec3<f32>(4.0 - 3.0 * st.x, 7.0, 1.0 + 3.0 * st.x);
    let u = vec3<f32>((3.0 - 2.0 * st.x) / uw.x - 2.0, (3.0 + st.x) / uw.y, st.x / uw.z + 2.0);
    let vw = vec3<f32>(4.0 - 3.0 * st.y, 7.0, 1.0 + 3.0 * st.y);
    let v = vec3<f32>((3.0 - 2.0 * st.y) / vw.x - 2.0, (3.0 + st.y) / vw.y, st.y / vw.z + 2.0);
    var sum = 0.0;
    for (var j = 0; j < 3; j++) {
        for (var i = 0; i < 3; i++) {
            let offset = vec2<f32>(u[i], v[j]) * texel;
            sum += uw[i] * vw[j]
                * textureSampleCompareLevel(shadow_map, shadow_sampler, base_uv + offset, layer, depth);
        }
    }
    return sum / 144.0;
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

fn classic() -> bool {
    return globals.weather.w > 0.5;
}

// Fixed shading per side, so the shape of the terrain reads in classic
// light: tops brightest, bottoms darkest.
fn face_shade(normal: vec3<f32>) -> f32 {
    return select(select(0.6, 0.8, normal.z != 0.0), select(0.5, 1.0, normal.y > 0.0), normal.y != 0.0);
}

// Classic light, as in block games without shaders: sky light and block
// light by level, a fixed shade per side, fog at the edge of the world.
fn shade_classic(in: ChunkVertex, texel: vec3<f32>, distance: f32) -> vec4<f32> {
    let level = brightness(in.light);
    let shade = face_shade(in.normal);
    // With shadows, part of sky light comes straight from the sun or moon.
    var direction = shade;
    if globals.light.w > 0.0 {
        let facing = max(dot(in.normal, globals.light.xyz), 0.0);
        var direct = 0.0;
        if facing > 0.0 {
            direct = facing * sunlit(in.position, in.normal, distance, facing);
        }
        direction = mix(shade, 0.5 * shade + 0.65 * direct, globals.light.w);
    }
    let sky = globals.classic[4].rgb * level.x * direction;
    let light = max(max(sky, level.yzw * shade), vec3<f32>(globals.classic[4].w * shade));
    var color = texel * light * mix(0.45, 1.0, in.occlusion);
    if in.glows == 1u {
        color = texel;
    }
    let fog = smoothstep(globals.fog.x, globals.fog.y, distance);
    return vec4<f32>(output(mix(color, globals.classic[3].rgb, fog)), 1.0);
}

fn shade(in: ChunkVertex) -> vec4<f32> {
    // Per pixel: far-away tops are up to a tile across, too big to take
    // their corners' distances.
    let distance = length(in.position);
    let texel = textureSample(block_textures, block_sampler, in.uv, in.layer).rgb;
    if classic() {
        return shade_classic(in, texel, distance);
    }
    var color: vec3<f32>;
    if in.glows == 1u {
        // Glowing blocks show brighter at night, but not blindingly.
        color = texel * GLOW * pow(globals.screen.z, -0.6);
    } else {
        let level = brightness(in.light);
        let occlusion = mix(0.45, 1.0, in.occlusion);
        let sky = sky_light(in.normal) * level.x;
        let blocks = level.yzw * BLOCK_LIGHT;
        let direct = direct_light(in.position, in.normal, distance, in.light.x);
        color = texel * (direct + (sky + blocks + globals.light_color.w) * occlusion);
    }
    return vec4<f32>(finish(with_fog(color, in.position, distance)), 1.0);
}

// Light from the sun or the moon on a face, shadows and clouds included.
// `sky` is the face's sky light level, from 0 to 1.
fn direct_light(position: vec3<f32>, normal: vec3<f32>, distance: f32, sky: f32) -> vec3<f32> {
    let toward = globals.light.xyz;
    let facing = dot(normal, toward);
    if facing <= 0.0 {
        return vec3<f32>(0.0);
    }
    // Without shadows, sky light says where the sun can't reach: under
    // overhangs, less of it; in caves, none.
    var visible = smoothstep(0.7, 1.0, sky);
    if globals.light.w > 0.0 {
        let shadow = sunlit(position, normal, distance, facing) * smoothstep(0.3, 0.6, sky);
        visible = mix(visible, shadow, globals.light.w);
    }
    return globals.light_color.rgb * facing * visible * cloud_shadow(position, toward, distance);
}

// Sky light on a face turned towards `normal`.
fn sky_light(normal: vec3<f32>) -> vec3<f32> {
    let a = globals.ambient;
    return max(a[0].rgb + a[1].rgb * normal.x + a[2].rgb * normal.y + a[3].rgb * normal.z, vec3<f32>(0.0));
}

// The sky's colour from one light, the sun or the moon, in `direction`.
// Below the horizon it is the air between the eye and the ground: as bright
// as the sky as far above, which is as far from the light.
fn sky_from(direction: vec3<f32>, toward: vec3<f32>, layer: f32) -> vec3<f32> {
    let flat_direction = normalize(direction.xz + vec2<f32>(1e-5, 0.0));
    let flat_light = normalize(toward.xz + vec2<f32>(1e-5, 0.0));
    let around = acos(clamp(dot(flat_direction, flat_light), -1.0, 1.0)) / PI;
    let up = asin(min(abs(direction.y), 1.0)) / (0.5 * PI);
    // The table is finer near the light and near the horizon.
    let size = vec2<f32>(SKY_AZIMUTHS, SKY_ELEVATIONS);
    let uv = (sqrt(vec2<f32>(around, up)) * (size - 1.0) + 0.5) / size;
    return textureSampleLevel(sky_table, smooth_sampler, vec3<f32>(uv, layer), 0.0).rgb;
}

// The sky's light coming from `direction`.
fn sky_radiance(direction: vec3<f32>) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    if globals.sky.x >= 0.0 {
        color += sky_from(direction, globals.sun.xyz, globals.sky.x);
    }
    if globals.sky.y >= 0.0 {
        color += sky_from(direction, -globals.sun.xyz, globals.sky.y) * globals.sky.z;
    }
    // An overcast sky is greyer.
    let grey = dot(color, LUMINANCE);
    return mix(color, vec3<f32>(grey), globals.weather.x * 0.5) * globals.weather.y;
}

// Distant things fade into the sky behind them.
fn with_fog(color: vec3<f32>, position: vec3<f32>, distance: f32) -> vec3<f32> {
    let amount = fog_amount(position, distance);
    return mix(color, sky_radiance(position / max(distance, 1e-4)), amount);
}

// How much of something at `position` the haze hides, and at the edge of
// the world, the fog that hides everything.
fn fog_amount(position: vec3<f32>, distance: f32) -> f32 {
    return max(haze(position, distance), smoothstep(globals.fog.x, globals.fog.y, distance));
}

// How much of something at `position` the haze hides: it is thicker low
// down.
fn haze(position: vec3<f32>, distance: f32) -> f32 {
    let eye = max(camera_height(), 0.0);
    let rise = (max(eye + position.y, 0.0) - eye) / HAZE_HEIGHT;
    var through = distance;
    if abs(rise) > 1e-3 {
        through = distance * (1.0 - exp(-rise)) / rise;
    }
    return 1.0 - exp(-globals.fog.w * exp(-eye / HAZE_HEIGHT) * through);
}

// Light becomes a colour on screen: exposure, then the AgX filmic curve
// (after Troy Sobotka's AgX and Benjamin Wrensch's fit of it), so bright
// light rolls off into white instead of clipping, and colours stay rich.
fn finish(light: vec3<f32>) -> vec3<f32> {
    let agx_in = mat3x3<f32>(
        0.842479062253094, 0.0423282422610123, 0.0423756549057051,
        0.0784335999999992, 0.878468636469772, 0.0784336,
        0.0792237451477643, 0.0791661274605434, 0.879142973793104,
    );
    let agx_out = mat3x3<f32>(
        1.19687900512017, -0.0528968517574562, -0.0529716355144438,
        -0.0980208811401368, 1.15190312990417, -0.0980434501171241,
        -0.0990297440797205, -0.0989611768448433, 1.15107367264116,
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var v = agx_in * max(light * globals.screen.z, vec3<f32>(1e-10));
    v = (clamp(log2(v), vec3<f32>(min_ev), vec3<f32>(max_ev)) - min_ev) / (max_ev - min_ev);
    let v2 = v * v;
    let v4 = v2 * v2;
    v = 15.5 * v4 * v2 - 40.14 * v4 * v + 31.96 * v4 - 6.868 * v2 * v + 0.4298 * v2 + 0.1191 * v
        - 0.00232;
    // A little more contrast and colour than plain AgX.
    let luma = dot(v, LUMINANCE);
    v = pow(max(v, vec3<f32>(0.0)), vec3<f32>(LOOK_POWER));
    v = luma + LOOK_SATURATION * (v - luma);
    v = clamp(agx_out * v, vec3<f32>(0.0), vec3<f32>(1.0));
    return output(pow(v, vec3<f32>(2.2)));
}

// A linear colour as the target wants it: as is for sRGB targets, which
// encode it themselves, encoded otherwise.
fn output(color: vec3<f32>) -> vec3<f32> {
    if globals.screen.w > 0.5 {
        return select(
            1.055 * pow(color, vec3<f32>(1.0 / 2.4)) - 0.055,
            color * 12.92,
            color <= vec3<f32>(0.0031308),
        );
    }
    return color;
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

// The classic sky: a gradient from the horizon up, coloured towards the
// sun at sunrise and sunset.
fn classic_sky(direction: vec3<f32>) -> vec3<f32> {
    let up = direction.y;
    let flat_direction = normalize(direction.xz + vec2<f32>(1e-5, 0.0));
    let flat_sun = normalize(globals.sun.xz + vec2<f32>(1e-5, 0.0));
    let sunward = max(dot(flat_direction, flat_sun), 0.0);
    let horizon = globals.classic[1].rgb + globals.classic[2].rgb * (0.2 + 0.8 * sunward * sunward);
    var color = mix(horizon, globals.classic[0].rgb, smoothstep(-0.05, 0.5, up));
    // The glow right around the sun.
    let toward_sun = max(dot(direction, globals.sun.xyz), 0.0);
    return color + globals.classic[2].rgb * pow(toward_sun, 24.0) * 0.5;
}

@fragment
fn sky_fragment(in: SkyVertex) -> @location(0) vec4<f32> {
    let far = globals.inverse_view_proj * vec4<f32>(in.screen, 1.0, 1.0);
    let direction = normalize(far.xyz / far.w);
    // Below the horizon, where nothing is drawn, the horizon goes on.
    let level = normalize(vec3<f32>(direction.x, max(direction.y, 0.0), direction.z));
    var color: vec3<f32>;
    if classic() {
        color = output(classic_sky(level));
    } else {
        color = finish(sky_radiance(level));
    }

    // Stars: one in a few hundred cells of a grid around the camera. They
    // are added on screen, whatever the exposure.
    let up = direction.y;
    if globals.sky.w > 0.0 && up > 0.0 {
        let scale = 150.0;
        let cell = floor(direction * scale);
        let chance = hash(cell);
        if chance > 0.995 {
            let jitter = vec3<f32>(hash(cell + 1.3), hash(cell + 2.7), hash(cell + 5.1)) - 0.5;
            let star = normalize((cell + 0.5 + jitter * 0.5) / scale);
            let spread = length(direction - star) * scale;
            let twinkle = (chance - 0.995) / 0.005;
            color += output(vec3<f32>(smoothstep(0.35, 0.0, spread) * (0.4 + 0.6 * twinkle))
                * globals.sky.w
                * globals.weather.y
                * clamp(up * 5.0, 0.0, 1.0));
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

// How bright the discs of the sun and the moon are.
const SUN_DISC = 4.0;
const MOON_DISC = 0.08;

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
        * globals.weather.z;
    return out;
}

@fragment
fn celestial_fragment(in: CelestialVertex) -> @location(0) vec4<f32> {
    let texel = textureSample(sky_textures, block_sampler, in.uv, in.body);
    if classic() {
        return vec4<f32>(output(texel.rgb), texel.a * in.alpha);
    }
    // The sun takes the colour of the air it shines through.
    let light = select(globals.sun_disc.rgb * SUN_DISC, vec3<f32>(MOON_DISC), in.body == 1u);
    return vec4<f32>(finish(texel.rgb * light), texel.a * in.alpha);
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

// Clouds. A cloud isn't geometry: `cloud_density` says how much cloud there
// is at a point, and clouds are drawn by marching rays through the layer, or
// in blocky cells as in classic block games. See `clouds.rs`.

// The textures, as in `clouds.rs`.
const PATCH_PERIOD = 4096.0;
const DETAIL_PERIOD = 96.0;
const OBSTACLE_SPAN = 4096.0;

fn cloud_bottom() -> f32 {
    return globals.clouds.z;
}

fn cloud_top() -> f32 {
    return globals.clouds.w;
}

// Heights in `cloud_obstacles` start here, as `OBSTACLE_BASE` in
// `clouds.rs`.
fn obstacle_base() -> f32 {
    return cloud_bottom() - 32.0;
}
// How much a block of the densest cloud dims light passing through it.
const CLOUD_DENSITY = 0.2;
// The most of the direct light a cloud's shadow takes away: clouds let some
// through, scattered.
const CLOUD_SHADOW = 0.75;
// Cloud shadows fade out from this far away to that.
const CLOUD_SHADOW_FADE = vec2<f32>(250.0, 900.0);
// Light scattered many times inside a thick cloud, beyond what the few
// octaves of `cloud_lighting` account for: sunlit cloud is brilliant white.
const CLOUD_BRIGHTNESS = 1.6;
// How far clouds are pushed aside by an obstacle per block it rises over a
// block of ground.
const PARTING = 3.0;
// Rays through the clouds stop this far away.
const CLOUD_REACH = 4000.0;
// Clouds are marched at this fraction of the screen's width and height, as
// in `world_pass.rs`.
const CLOUD_SCALE = 4;

fn camera_height() -> f32 {
    return f32(globals.camera_block.y) + globals.camera_fract.y;
}

// How high obstacles near `xz` (relative to the camera) reach, as clouds
// keep clear of them.
fn obstacle_top(xz: vec2<f32>) -> f32 {
    let uv = (xz - globals.cloud_place.yz) / OBSTACLE_SPAN;
    let inside = all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
    let height = textureSampleLevel(cloud_obstacles, smooth_sampler, uv, 0.0).r * 255.0;
    return obstacle_base() + select(0.0, height, inside);
}

// How much cloud the sky holds over `xz`, from 0 at the edge of a patch to
// 1 deep inside it. With `part`, near an obstacle, the cloud there is the one
// that would have been closer to it: patches part around obstacles.
fn cloud_cover(xz: vec2<f32>, top: f32, part: bool) -> f32 {
    var push = vec2<f32>(0.0);
    if part && top > cloud_bottom() - 24.0 {
        let e = 4.0;
        let slope = vec2<f32>(
            obstacle_top(xz + vec2<f32>(e, 0.0)) - obstacle_top(xz - vec2<f32>(e, 0.0)),
            obstacle_top(xz + vec2<f32>(0.0, e)) - obstacle_top(xz - vec2<f32>(0.0, e)),
        ) / (2.0 * e);
        push = slope * PARTING;
    }
    let uv = globals.cloud_offset.xy + (xz + push) / PATCH_PERIOD;
    let spot = textureSampleLevel(cloud_patches, wrap_sampler, uv, 0.0).r;
    // Patches thicken towards their middles; under an overcast sky, they
    // are thick almost everywhere.
    let cover = globals.clouds.x;
    return saturate((spot - (1.0 - cover)) / (0.05 + 0.25 * (1.0 - cover)));
}

// The puff noise at `position` (relative to the camera), at `scale` times
// the size of `cloud_detail`; it moves with the wind.
fn puffs(position: vec3<f32>, scale: f32) -> f32 {
    let at = vec3<f32>(globals.cloud_offset.z, globals.cloud_place.x, globals.cloud_offset.w)
        * (1.0 / scale) + position / (DETAIL_PERIOD * scale);
    return textureSampleLevel(cloud_detail, wrap_sampler, at, 0.0).r;
}

// How dense the cloud is at `position` (relative to the camera), from 0 to
// 1. `detail`, from 0 to 1, is how much its edges are worn into small puffs:
// none far away, where they would only flicker. Without `part`, a little
// cheaper: patches don't part around obstacles, only thin out near them,
// which is enough for shadows.
fn cloud_density(position: vec3<f32>, detail: f32, part: bool) -> f32 {
    let y = position.y + camera_height();
    let h = (y - cloud_bottom()) / (cloud_top() - cloud_bottom());
    if h <= 0.0 || h >= 1.0 {
        return 0.0;
    }
    let top = obstacle_top(position.xz);
    let cover = cloud_cover(position.xz, top, part);
    if cover <= 0.0 {
        return 0.0;
    }
    // Flat bottoms and round tops; the deeper into a patch, the taller.
    let height = 0.3 + 0.7 * cover;
    let profile = smoothstep(0.0, 0.07, h) * (1.0 - smoothstep(height * 0.35, height, h));
    // Big billows carve the shape out of that.
    let billows = puffs(position, 3.0);
    var density = saturate(cover * profile * 1.8 - (1.0 - billows) * 0.9);
    // Nothing near the top of an obstacle; its sides are kept clear by the
    // widened obstacle map.
    density *= smoothstep(top + 1.0, top + 8.0, y);
    if detail > 0.0 && density > 0.0 && density < 0.6 {
        // Small puffs wear the thin edges away.
        let wear = (1.0 - puffs(position, 1.0)) * 0.3 * detail;
        density = saturate((density - wear) / (1.0 - wear));
    }
    return density;
}

// Whether the clouds are blocky rather than soft.
fn blocky_clouds() -> bool {
    return globals.cloud_shadow.w > 0.5;
}

// The bottom of the clouds that cast shadows.
fn shadow_bottom() -> f32 {
    return select(cloud_bottom(), globals.blocky.z, blocky_clouds());
}

// How much of the light gets through the clouds to `position`, `distance`
// from the eye, on its way along `toward`.
fn cloud_shadow(position: vec3<f32>, toward: vec3<f32>, distance: f32) -> f32 {
    if globals.clouds.y == 0.0 || toward.y < 0.02 {
        return 1.0;
    }
    // Where the light on its way here enters the bottom of the clouds.
    let rise = max(shadow_bottom() - camera_height() - position.y, 0.0);
    let entry = position.xz + toward.xz * (rise / toward.y);
    let uv = (entry - globals.cloud_shadow.xy) / globals.cloud_shadow.z;
    let through = textureSampleLevel(cloud_shadows, smooth_sampler, uv, 0.0).r;
    // Clouds let some light through, scattered. Far away, the haze and the
    // light the clouds themselves scatter wash their shadows out; low light
    // reaches under them from the side.
    let strength = CLOUD_SHADOW
        * (1.0 - smoothstep(CLOUD_SHADOW_FADE.x, CLOUD_SHADOW_FADE.y, distance))
        * smoothstep(0.02, 0.15, toward.y);
    return 1.0 - (1.0 - through) * strength;
}

// The cloud shadow map: for each place in it, how much light gets through
// the clouds along the light, from where it enters their bottom.
@fragment
fn cloud_shadow_fragment(in: SkyVertex) -> @location(0) vec4<f32> {
    let uv = vec2<f32>(in.screen.x * 0.5 + 0.5, 0.5 - in.screen.y * 0.5);
    let entry = globals.cloud_shadow.xy + uv * globals.cloud_shadow.z;
    let toward = globals.light.xyz;
    if blocky_clouds() {
        return vec4<f32>(blocky_shadow(entry, toward));
    }
    let length = (cloud_top() - cloud_bottom()) / max(toward.y, 0.05);
    let bottom = vec3<f32>(entry.x, cloud_bottom() - camera_height(), entry.y);
    var density = 0.0;
    for (var i = 0; i < 4; i++) {
        density += cloud_density(bottom + toward * (length * (f32(i) + 0.5) / 4.0), 0.0, false);
    }
    return vec4<f32>(exp(-density * length / 4.0 * CLOUD_DENSITY));
}

// How much light gets through the blocky clouds from where it enters the
// bottom of their layer at `entry` (x and z relative to the camera), on its
// way along `toward`: none where it crosses a cloud.
fn blocky_shadow(entry: vec2<f32>, toward: vec3<f32>) -> f32 {
    let across = toward.xz * (globals.blocky.w / max(toward.y, 0.05));
    var cloud = 0.0;
    for (var i = 0; i < 4; i++) {
        let along = (f32(i) + 0.5) / 4.0;
        let xz = entry + across * along;
        let cell = vec2<i32>(floor((globals.blocky.xy + xz) / BLOCKY_CELL));
        if blocky_cell(cell) {
            // Thinning out near the tops of obstacles, as the clouds do.
            let obstacle = obstacle_top(xz);
            let y = globals.blocky.z + globals.blocky.w * along;
            cloud += smoothstep(obstacle + 1.0, obstacle + 8.0, y);
        }
    }
    return 1.0 - cloud / 4.0;
}

// Henyey-Greenstein's phase function: how much light scatters by an angle.
fn henyey_greenstein(cos_theta: f32, g: f32) -> f32 {
    let g2 = g * g;
    return (1.0 - g2) / (4.0 * PI * pow(1.0 + g2 - 2.0 * g * cos_theta, 1.5));
}

// Clouds scatter light mostly onwards, which lights their edges against the
// sun, and a little back. `spread` below 1 evens it out, for light that has
// scattered many times.
fn cloud_phase(cos_theta: f32, spread: f32) -> f32 {
    return mix(
        henyey_greenstein(cos_theta, -0.2 * spread),
        henyey_greenstein(cos_theta, 0.75 * spread),
        0.7,
    );
}

// Light reaching a point in a cloud: from the clouds' light, dimmed by the
// cloud on its way (`depth`), scattered towards the eye, as if scattered a
// few times (after Wrenninge's approximation); and from the sky, more at the
// top of the layer than at the bottom.
fn cloud_lighting(depth: f32, cos_theta: f32, h: f32) -> vec3<f32> {
    // Seen with the light behind the eye, the edges of a cloud are darker
    // than its depths, where light has scattered back out ("powder").
    let powder = mix(1.0 - exp(-depth * 2.0), 1.0, saturate(cos_theta * 0.5 + 0.5));
    var direct = 0.0;
    var strength = 1.0;
    var reach = 1.0;
    var spread = 1.0;
    for (var octave = 0; octave < 4; octave++) {
        direct += strength * exp(-depth * reach) * cloud_phase(cos_theta, spread);
        strength *= 0.7;
        reach *= 0.35;
        spread *= 0.5;
    }
    let a = globals.ambient;
    let sky = max(a[0].rgb + a[2].rgb * mix(-0.6, 1.0, h), vec3<f32>(0.0));
    return (globals.cloud_light_color.rgb * PI * direct * powder + sky * mix(0.25, 0.6, h))
        * CLOUD_BRIGHTNESS;
}

// How much cloud lies between `position` and the clouds' light.
fn cloud_depth_towards_light(position: vec3<f32>) -> f32 {
    let toward = globals.cloud_light.xyz;
    var depth = 0.0;
    var t = 0.0;
    var step = 3.0;
    for (var i = 0; i < 4; i++) {
        depth += cloud_density(position + toward * (t + step * 0.5), 0.0, false) * step;
        t += step;
        step *= 2.0;
    }
    return depth * CLOUD_DENSITY;
}

// A different, patternless number from 0 to 1 for each pixel, so rays don't
// step in lockstep and band.
fn pixel_noise(pixel: vec2<f32>) -> f32 {
    var h = vec2<u32>(pixel);
    var n = h.x * 1597334677u ^ h.y * 3812015801u;
    n = (n ^ (n >> 16u)) * 2246822519u;
    n = n ^ (n >> 13u);
    return f32(n) / 4294967296.0;
}

fn view_direction(screen: vec2<f32>) -> vec3<f32> {
    let far = globals.inverse_view_proj * vec4<f32>(screen, 1.0, 1.0);
    return normalize(far.xyz / far.w);
}

struct CloudMarch {
    // rgb: light scattered towards the eye; a: how much of what's behind
    // shows through.
    @location(0) color: vec4<f32>,
    // How far away the clouds start along the ray.
    @location(1) distance: f32,
}

// The depth buffer of the world just drawn, at full resolution.
@group(1) @binding(0) var scene_depth: texture_depth_2d;

// Marches a ray through the clouds for each pixel of a target a quarter of
// the screen's width and height.
@fragment
fn cloud_march_fragment(in: SkyVertex) -> CloudMarch {
    // Each frame, one pixel of every 2×2 square of the clouds is marched,
    // drawn packed together into a target half as wide and tall; the others
    // come from earlier frames (`cloud_resolve_fragment`).
    let size = vec2<i32>(textureDimensions(scene_depth));
    let clouds = (size + CLOUD_SCALE - 1) / CLOUD_SCALE;
    let turn = i32(globals.cloud_past.w);
    var pixel = vec2<i32>(in.clip.xy);
    if turn >= 0 {
        pixel = pixel * 2 + vec2<i32>(turn & 1, turn >> 1);
    }
    let screen = (vec2<f32>(pixel) + 0.5) / vec2<f32>(clouds) * vec2<f32>(2.0, -2.0)
        + vec2<f32>(-1.0, 1.0);
    let direction = view_direction(screen);
    // How far the world lets the ray go: the farthest of the corners of the
    // pixels this one covers, so clouds reach the edges of what's in front.
    let corner = pixel * CLOUD_SCALE;
    var depth = 0.0;
    for (var i = 0; i < 4; i++) {
        let offset = vec2<i32>(i & 1, i >> 1) * (CLOUD_SCALE - 1);
        depth = max(depth, textureLoad(scene_depth, min(corner + offset, size - 1), 0));
    }
    var limit = CLOUD_REACH;
    if depth < 1.0 {
        let point = globals.inverse_view_proj * vec4<f32>(screen, depth, 1.0);
        limit = min(limit, length(point.xyz / point.w));
    }

    var out: CloudMarch;
    out.color = vec4<f32>(0.0, 0.0, 0.0, 1.0);
    out.distance = CLOUD_REACH;
    // Where the ray is inside the layer.
    let bottom = cloud_bottom() - camera_height();
    let top = cloud_top() - camera_height();
    var start = 0.0;
    var end = limit;
    if abs(direction.y) > 1e-4 {
        let a = bottom / direction.y;
        let b = top / direction.y;
        start = max(min(a, b), 0.0);
        end = min(max(a, b), limit);
    } else if bottom > 0.0 || top < 0.0 {
        end = 0.0;
    }
    if end <= start {
        return out;
    }

    // Short steps near the eye and inside clouds, longer far away, and longer
    // still through clear air.
    let cos_theta = dot(direction, globals.cloud_light.xyz);
    let jitter = pixel_noise(vec2<f32>(pixel));
    var t = start;
    var light = vec3<f32>(0.0);
    var through = 1.0;
    var entry = -1.0;
    var clear = 0.0;
    var towards_light = 0.0;
    // Never shorter than a sixtieth of the way through the layer.
    let shortest = max((end - start) / 60.0, 1.0);
    for (var i = 0; i < 64; i++) {
        var dt = clamp(t * 0.012, shortest, 32.0) * (1.0 + clear);
        if entry >= 0.0 && clear == 0.0 {
            dt = clamp(t * 0.008, shortest, 16.0);
        }
        let position = direction * (t + dt * jitter);
        let density = cloud_density(position, 1.0 - smoothstep(100.0, 400.0, t), true);
        if density > 0.001 {
            if entry < 0.0 {
                entry = t;
            }
            let h = (position.y - bottom) / (top - bottom);
            // Deep in, little of the light gets out to the eye: the last
            // way to the light will do.
            if through > 0.25 {
                towards_light = cloud_depth_towards_light(position);
            }
            let lit = cloud_lighting(towards_light, cos_theta, h);
            let step_through = exp(-density * CLOUD_DENSITY * dt);
            light += through * lit * (1.0 - step_through);
            through *= step_through;
            if through < 0.01 {
                break;
            }
            clear = 0.0;
        } else {
            clear = min(clear + 1.0, 3.0);
        }
        t += dt;
        if t >= end {
            break;
        }
    }
    if entry < 0.0 {
        return out;
    }
    // Far clouds sink into the haze. They go on past the edge of the
    // world, out to the horizon.
    let hidden = haze(direction * entry, entry);
    light = mix(light, (1.0 - through) * sky_radiance(direction), hidden);
    out.color = vec4<f32>(light, through);
    out.distance = entry;
    return out;
}

// The clouds marched this frame, for some pixels, and the clouds as they
// were last frame, for all of them.
@group(1) @binding(3) var fresh_color: texture_2d<f32>;
@group(1) @binding(4) var fresh_distance: texture_2d<f32>;
@group(1) @binding(5) var past_color: texture_2d<f32>;
@group(1) @binding(6) var past_distance: texture_2d<f32>;

// The clouds for every pixel: marched this frame where they were, and
// elsewhere where they were last frame, moved along with the camera and the
// wind. Where that is off the screen, the pixel marched nearby will do.
@fragment
fn cloud_resolve_fragment(in: SkyVertex) -> CloudMarch {
    let pixel = vec2<i32>(in.clip.xy);
    let turn = globals.cloud_past.w;
    var out: CloudMarch;
    if turn < 0.0 {
        out.color = textureLoad(fresh_color, pixel, 0);
        out.distance = textureLoad(fresh_distance, pixel, 0).r;
        return out;
    }
    // The pixel of this 2×2 square marched this frame, and where it was
    // drawn: packed together at the start of the target.
    let size = vec2<i32>(textureDimensions(fresh_color));
    let marched = (pixel / 2) * 2 + vec2<i32>(i32(turn) & 1, i32(turn) >> 1);
    let packed = min(pixel / 2, (size + 1) / 2 - 1);
    out.color = textureLoad(fresh_color, packed, 0);
    out.distance = textureLoad(fresh_distance, packed, 0).r;
    if all(pixel == marched) {
        return out;
    }
    let point = view_direction(in.screen) * out.distance + globals.cloud_past.xyz;
    let clip = globals.cloud_past_view_proj * vec4<f32>(point, 1.0);
    let uv = vec2<f32>(clip.x, -clip.y) / clip.w * 0.5 + 0.5;
    if clip.w > 0.0 && all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) {
        out.color = textureSampleLevel(past_color, smooth_sampler, uv, 0.0);
        let texel = clamp(vec2<i32>(uv * vec2<f32>(size)), vec2<i32>(0), size - 1);
        out.distance = textureLoad(past_distance, texel, 0).r;
    }
    return out;
}

// The clouds for every pixel, at a quarter of the size.
@group(1) @binding(1) var cloud_color: texture_2d<f32>;
@group(1) @binding(2) var cloud_distance: texture_2d<f32>;

struct CloudComposite {
    @location(0) color: vec4<f32>,
    // Where the clouds start, so the world in front of them hides them.
    @builtin(frag_depth) depth: f32,
}

// Lays the marched clouds over the world: the target keeps `a` of what it
// had, and gets `rgb` added.
@fragment
fn cloud_composite_fragment(in: SkyVertex) -> CloudComposite {
    let size = vec2<f32>(textureDimensions(cloud_color));
    let uv = in.clip.xy / globals.screen.xy;
    let middle = textureSampleLevel(cloud_color, smooth_sampler, uv, 0.0);
    // The nearest of the four texels around: where clouds are in front of
    // the world, they show over it, up to the edges.
    let corner = vec2<i32>(uv * size - 0.5);
    var distance = CLOUD_REACH;
    for (var i = 0; i < 4; i++) {
        let texel = clamp(corner + vec2<i32>(i & 1, i >> 1), vec2<i32>(0), vec2<i32>(size) - 1);
        distance = min(distance, textureLoad(cloud_distance, texel, 0).r);
    }
    let point = view_direction(in.screen) * distance;
    let clip = globals.view_proj * vec4<f32>(point, 1.0);

    var out: CloudComposite;
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    // Four more filtered taps around it smooth the grain of the marching.
    let spread = 1.0 / size;
    let around = textureSampleLevel(cloud_color, smooth_sampler, uv + vec2<f32>(spread.x, spread.y * 0.5), 0.0)
        + textureSampleLevel(cloud_color, smooth_sampler, uv - vec2<f32>(spread.x, spread.y * 0.5), 0.0)
        + textureSampleLevel(cloud_color, smooth_sampler, uv + vec2<f32>(-spread.x * 0.5, spread.y), 0.0)
        + textureSampleLevel(cloud_color, smooth_sampler, uv + vec2<f32>(spread.x * 0.5, -spread.y), 0.0);
    let marched = (middle + around) / 5.0;
    let cover = 1.0 - marched.a;
    out.color = vec4<f32>(finish(marched.rgb / max(cover, 1e-4)) * cover, marched.a);
    return out;
}

// Clouds drawn over the screen, and how far away they are.
struct ScreenClouds {
    @location(0) color: vec4<f32>,
    // So the world in front of them hides them.
    @builtin(frag_depth) depth: f32,
}

// Whether a cell of blocky clouds holds a cloud: where its middle lies in a
// patch of cloud, the same patches the soft clouds are made of.
fn blocky_cell(cell: vec2<i32>) -> bool {
    let middle = (vec2<f32>(cell) + 0.5) * BLOCKY_CELL;
    let spot = textureSampleLevel(cloud_patches, wrap_sampler, middle / PATCH_PERIOD, 0.0).r;
    return spot > 1.0 - globals.clouds.x;
}

// Blocky clouds, as in classic block games: cells of `BLOCKY_CELL` blocks in
// a thin layer. Drawn over the screen, each pixel walks its ray across the
// cells of the layer until it meets a cloud.
@fragment
fn blocky_cloud_fragment(in: SkyVertex) -> ScreenClouds {
    let direction = view_direction(in.screen);
    let bottom = globals.blocky.z - camera_height();
    let top = bottom + globals.blocky.w;
    let reach = max(globals.fog.y, 512.0);
    var start = 0.0;
    var end = reach;
    if abs(direction.y) > 1e-5 {
        let a = bottom / direction.y;
        let b = top / direction.y;
        start = max(min(a, b), 0.0);
        end = min(max(a, b), reach);
    } else if bottom > 0.0 || top < 0.0 {
        discard;
    }
    if end <= start {
        discard;
    }

    // Walk across the cells, along x and z.
    let flat = direction.xz;
    let begin = globals.blocky.xy + flat * start;
    var cell = vec2<i32>(floor(begin / BLOCKY_CELL));
    let step = vec2<i32>(select(vec2<f32>(-1.0), vec2<f32>(1.0), flat >= vec2<f32>(0.0)));
    let across = BLOCKY_CELL / max(abs(flat), vec2<f32>(1e-6));
    let first_edge = (vec2<f32>(cell) + select(vec2<f32>(0.0), vec2<f32>(1.0), flat >= vec2<f32>(0.0)))
        * BLOCKY_CELL;
    var next = start + (first_edge - begin) / select(flat, vec2<f32>(1e-6), abs(flat) < vec2<f32>(1e-6));
    next = select(next, vec2<f32>(1e9), abs(flat) < vec2<f32>(1e-6));
    var t = start;
    // Entered through the bottom or the top, or a side along x or z.
    var side = 0;
    var hit = false;
    for (var i = 0; i < 160; i++) {
        if blocky_cell(cell) {
            hit = true;
            break;
        }
        if next.x < next.y {
            t = next.x;
            next.x += across.x;
            cell.x += step.x;
            side = 1;
        } else {
            t = next.y;
            next.y += across.y;
            cell.y += step.y;
            side = 2;
        }
        if t >= end {
            break;
        }
    }
    if !hit {
        discard;
    }
    var normal = vec3<f32>(0.0, -sign(direction.y), 0.0);
    if side == 1 {
        normal = vec3<f32>(-f32(step.x), 0.0, 0.0);
    } else if side == 2 {
        normal = vec3<f32>(0.0, 0.0, -f32(step.y));
    } else if start == 0.0 {
        // Inside a cloud.
        normal = -direction;
    }
    let position = direction * max(t, 0.05);
    // Never inside buildings and mountains: thinning out near their tops.
    let obstacle = obstacle_top(position.xz);
    let clear = smoothstep(obstacle + 1.0, obstacle + 8.0, position.y + camera_height());
    let fade = (1.0 - smoothstep(reach * 0.6, reach, t)) * clear;
    if fade <= 0.0 {
        discard;
    }
    var color: vec3<f32>;
    if classic() {
        // Lit from above, darker underneath.
        var shade = select(0.86, 1.0, normal.y > 0.5);
        shade = select(shade, 0.7, normal.y < -0.5);
        let fog = smoothstep(globals.fog.x, globals.fog.y, t);
        color = output(mix(globals.classic[5].rgb * shade, globals.classic[3].rgb, fog));
    } else {
        let facing = dot(normal, globals.cloud_light.xyz) * 0.75 + 0.25;
        let lit = globals.cloud_light_color.rgb * max(facing, 0.0) + sky_light(normal);
        color = finish(mix(CLOUD_ALBEDO * lit, sky_radiance(direction), haze(position, t)));
    }
    var out: ScreenClouds;
    out.color = vec4<f32>(color, 0.8 * fade);
    let clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}
