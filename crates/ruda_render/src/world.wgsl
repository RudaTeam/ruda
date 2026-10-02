// The sky, the block world, the outline of the targeted block and the
// crosshair.
//
// Chunks are drawn as packed quads (see `mesh.rs`), each stored once per
// corner: a vertex is a whole quad, and which corner it is comes from its
// index, so the vertex shader works out where that corner lies. Bits 18 to
// 31 of the quad's second word are its chunk's slot (see `world_pass.rs`).

struct Globals {
    view_proj: mat4x4<f32>,
    // Turns screen positions back into directions, for the sky.
    inverse_view_proj: mat4x4<f32>,
    // x: distance where fog starts hiding the edge of the world, y: where it
    // hides everything, w: haze per block at sea level; z: 1 with shadows on,
    // also while they fade out with the light sinking to the horizon.
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
    shadow_view_proj: array<mat4x4<f32>, 4>,
    // How far each cascade reaches. The last one shows far-away terrain;
    // when it reaches no farther than the one before, there is none.
    shadow: vec4<f32>,
    // x: how much of the sky clouds cover; y: how dense they are, 1 for
    // fair-weather clouds, 0 without clouds; z,
    // w: the bottom and the top of the cloud layer.
    clouds: vec4<f32>,
    // xy: where the camera is over `cloud_cells`, in blocks: it moves with
    // the wind, and repeats with the map; zw: the first corner of
    // `cloud_obstacles`, x and z relative to the camera.
    cloud_place: vec4<f32>,
    // xyz: towards the light the clouds get: high up, they still see the
    // sun for a while after it has set on the ground.
    cloud_light: vec4<f32>,
    // rgb: that light on a white surface facing it.
    cloud_light_color: vec4<f32>,
    // The layer of `shadow_map` each cascade is shown from.
    shadow_layers: vec4<f32>,
    // Classic light (see `ClassicSky` in `sky.rs`): sky colour straight up
    // (w: how bright the stars are), at the horizon, the glow of sunrise and
    // sunset, the fog, sky light (w: the least light anything gets) and a
    // cloud's lit top.
    classic: array<vec4<f32>, 6>,
    // Rays towards the light: x, z: the first block of `ray_solids` along x
    // and z; y: its lowest block; w: how high above that solid blocks reach,
    // 0 without rays.
    ray_window: vec4<i32>,
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
// The same map read as plain numbers: OpenGL can only compare depth
// textures, not read them.
@group(0) @binding(11) var shadow_depths: texture_2d_array<f32>;
// For each square of `SHADOW_HINT_BLOCK` texels of the far cascade's shown
// layer, the depth nearest the light in it and the squares around it, as
// bits of a float; see `hint_blocks_fragment` and `hint_fragment`.
@group(0) @binding(19) var shadow_hint: texture_2d<u32>;
const SHADOW_HINT_SIZE = 128;
const SHADOW_HINT_BLOCK = 16;
// Which cells of the layer hold cloud, 1 for those, repeating; see
// `cloud_cells` in `clouds.rs`.
@group(0) @binding(7) var cloud_cells: texture_2d<f32>;
// Linear filtering, clamped at the edges.
@group(0) @binding(8) var smooth_sampler: sampler;
// Linear filtering, repeating.
@group(0) @binding(10) var wrap_sampler: sampler;
// How high obstacles reach into the clouds around the camera, widened and
// softened, above `OBSTACLE_BASE`.
@group(0) @binding(12) var cloud_obstacles: texture_2d<f32>;
// Which chunks are drawn in full this frame: a texel per column of chunks,
// repeating every `DRAWN_WIDTH` columns, and in it a bit per chunk up the
// column, repeating every 32.
@group(0) @binding(14) var chunks_drawn: texture_2d<u32>;
const DRAWN_WIDTH = 128;
// The world near the camera, for rays towards the light (see `rays.rs`): a
// word of 32 blocks up y for each x and z, a layer per layer of chunks,
// wrapping around every `RAY_WINDOW` blocks along x and z; and for each brick
// of `RAY_BRICK`³ blocks, whether it holds a solid one.
@group(0) @binding(16) var ray_solids: texture_3d<u32>;
@group(0) @binding(17) var ray_bricks: texture_3d<u32>;
// How high solid blocks reach in each column of bricks, from the window's
// lowest block; level 2 per column of chunks, level 4 per square of 4 × 4
// of them.
@group(0) @binding(18) var ray_tops: texture_2d<u32>;
const RAY_WINDOW = 512;
const RAY_BRICK = 8.0;
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
    let corner = quad_corner(quad, vertex & 3u);
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
    let corner = quad_corner(quad, vertex & 3u);
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

// Whether `block` of the ray window (from its first corner) is a solid cube.
fn ray_solid(block: vec3<i32>) -> bool {
    let at = block + vec3<i32>(globals.ray_window.x, 0, globals.ray_window.z);
    let word = textureLoad(ray_solids, vec3<i32>(at.x & (RAY_WINDOW - 1), at.y >> 5u, at.z & (RAY_WINDOW - 1)), 0).r;
    return ((word >> u32(at.y & 31)) & 1u) != 0u;
}

// Whether `brick` of the ray window holds a solid cube.
fn ray_brick(brick: vec3<i32>) -> bool {
    let at = brick + vec3<i32>(globals.ray_window.x, 0, globals.ray_window.z) / 8;
    let size = RAY_WINDOW / 8;
    return textureLoad(ray_bricks, vec3<i32>(at.x & (size - 1), at.y, at.z & (size - 1)), 0).r != 0u;
}

// How far a ray from `origin`, along the direction whose reciprocal is
// `inverse`, goes until it leaves the box from `low` to `high`.
fn slab_exit(origin: vec3<f32>, inverse: vec3<f32>, low: vec3<f32>, high: vec3<f32>) -> f32 {
    let far = select(low, high, inverse > vec3<f32>(0.0));
    let t = (far - origin) * inverse;
    return min(min(t.x, t.y), t.z);
}

// How high solid blocks reach in `column` of bricks of the ray window.
fn ray_top(column: vec2<i32>) -> f32 {
    let at = column + vec2<i32>(globals.ray_window.x, globals.ray_window.z) / 8;
    let size = RAY_WINDOW / 8;
    return f32(textureLoad(ray_tops, vec2<i32>(at.x & (size - 1), at.y & (size - 1)), 0).r);
}

// How high solid blocks reach in the region of `8 << level` blocks across
// holding `block` (x and z from the window's first block): a column of
// chunks for level 2, a square of 4 × 4 of them for level 4.
fn ray_region_top(block: vec2<i32>, level: u32) -> f32 {
    let at = block + vec2<i32>(globals.ray_window.x, globals.ray_window.z);
    let size = (RAY_WINDOW / 8) >> level;
    let region = at >> vec2<u32>(3u + level);
    return f32(textureLoad(ray_tops, region & vec2<i32>(size - 1), i32(level)).r);
}

// A ray from `position` (on a face turned towards `normal`) towards the
// light through the solid cubes near the camera: x is 1 if it gets out of
// the window of the world the GPU holds without meeting one, 0 if it meets
// one; y is how far it went. The light is above the horizon, so the ray only
// rises: above all the solid blocks of a square of chunks, or of a column of
// them, it crosses it in one step; otherwise it crosses columns of bricks,
// skipping those whose solid blocks it is already above, and inside the
// others bricks, skipping those with none, then blocks.
fn ray_shadow(position: vec3<f32>, normal: vec3<f32>) -> vec2<f32> {
    let window = globals.ray_window;
    let toward = globals.light.xyz;
    let first = vec3<i32>(window.x, window.y, window.z);
    // Above `top`, nothing is solid.
    let top = f32(window.w);
    let start = position + globals.camera_fract.xyz + vec3<f32>(globals.camera_block.xyz - first)
        + normal * 0.002;
    let size = vec3<f32>(f32(RAY_WINDOW), top, f32(RAY_WINDOW));
    if toward.y <= 0.0 || any(start.xz < vec2<f32>(0.0)) || any(start.xz >= size.xz)
        || start.y < 0.0 {
        return vec2<f32>(1.0, 0.0);
    }
    if start.y >= top {
        return vec2<f32>(1.0, 0.0);
    }
    let inverse = 1.0 / select(toward, vec3<f32>(1e-6), abs(toward) < vec3<f32>(1e-6));
    let out = slab_exit(start, inverse, vec3<f32>(0.0), size);
    var t = 0.0;
    for (var i = 0; i < 160; i++) {
        let at = start + toward * (t + 1e-3);
        // Over a whole region: on to where the ray leaves it.
        let block = vec2<i32>(floor(at.xz));
        var level = 4u;
        var over = at.y >= ray_region_top(block, level);
        if !over {
            level = 2u;
            over = at.y >= ray_region_top(block, level);
        }
        if over {
            let span = 8 << level;
            let first_block = ((block + window.xz) & vec2<i32>(-span)) - window.xz;
            let low = vec3<f32>(f32(first_block.x), -1e6, f32(first_block.y));
            t = min(slab_exit(start, inverse, low, low + vec3<f32>(f32(span), 2e6, f32(span))), out);
            if t >= out {
                break;
            }
            continue;
        }
        let column = vec2<i32>(floor(at.xz / RAY_BRICK));
        let corner = vec3<f32>(vec2<f32>(column).x * RAY_BRICK, -1e6, vec2<f32>(column).y * RAY_BRICK);
        let leave = min(slab_exit(start, inverse, corner, corner + vec3<f32>(RAY_BRICK, 2e6, RAY_BRICK)), out);
        let column_top = ray_top(column);
        if at.y < column_top {
            // Up to where the ray rises above the column's solid blocks.
            let until = min(leave, (column_top - start.y) * inverse.y);
            var u = t;
            for (var j = 0; j < 64; j++) {
                let brick = vec3<i32>(floor((start + toward * (u + 1e-3)) / RAY_BRICK));
                let low = vec3<f32>(brick) * RAY_BRICK;
                let past = min(slab_exit(start, inverse, low, low + RAY_BRICK), until);
                if ray_brick(brick) {
                    // Block by block across the brick.
                    var s = u;
                    for (var k = 0; k < 24; k++) {
                        let block = vec3<i32>(floor(start + toward * (s + 1e-3)));
                        if ray_solid(block) {
                            return vec2<f32>(0.0, s);
                        }
                        let cell = vec3<f32>(block);
                        s = slab_exit(start, inverse, cell, cell + 1.0);
                        if s >= past {
                            break;
                        }
                    }
                }
                u = past;
                if u >= until {
                    break;
                }
            }
        }
        t = leave;
        if t >= out {
            break;
        }
    }
    return vec2<f32>(1.0, t);
}

// How much of the sun's (or moon's) direct light reaches a point: 0 in full
// shadow, 1 in full light. Each near cascade gives way to the next over the
// last fifth of its reach, and the last of them fades out; the far cascade
// adds the shadows of far-away terrain, near the camera only of what lies
// farther towards the light than the near cascades see.
fn sunlit(position: vec3<f32>, normal: vec3<f32>, distance: f32, facing: f32) -> f32 {
    let reach = globals.shadow;
    // With rays: what they meet near the camera, and farther along them,
    // far-away terrain from the far cascade.
    if globals.ray_window.w > 0 {
        let ray = ray_shadow(position, normal);
        var lit = ray.x;
        if lit > 0.0 && reach.w > reach.z && distance < reach.w {
            let far = cascade_shadow(3, position, normal, facing, max(ray.y, FAR_SHADOW_LIFT.x));
            lit *= mix(far, 1.0, smoothstep(reach.w * 0.8, reach.w, distance));
        }
        return lit;
    }
    var lit = 1.0;
    // How much of the near cascades counts here.
    var near = 0.0;
    if distance < reach.z {
        var cascade = 2;
        if distance < reach.y {
            cascade = 1;
        }
        if distance < reach.x {
            cascade = 0;
        }
        lit = cascade_shadow(cascade, position, normal, facing, 0.0);
        let blend = smoothstep(reach[cascade] * 0.8, reach[cascade], distance);
        if blend > 0.0 {
            var next = 1.0;
            if cascade < 2 {
                next = cascade_shadow(cascade + 1, position, normal, facing, 0.0);
            }
            lit = mix(lit, next, blend);
        }
        near = 1.0 - smoothstep(reach.z * 0.8, reach.z, distance);
    }
    if reach.w > reach.z && distance < reach.w {
        let lift = mix(FAR_SHADOW_LIFT.x, FAR_SHADOW_LIFT.y, near);
        let far = cascade_shadow(3, position, normal, facing, lift);
        lit *= mix(far, 1.0, smoothstep(reach.w * 0.8, reach.w, distance));
    }
    return lit;
}

// Room the cascades see towards the light, as `CASTER_MARGIN` and
// `FAR_CASTER_MARGIN` in `shadows.rs`.
const CASTER_MARGIN = 256.0;
const FAR_CASTER_MARGIN = 2048.0;

// The far cascade counts only what lies farther than this towards the
// light: everywhere, `x`, as far-away terrain is drawn coarse and would
// shade itself, and small things that near would only blur; near the
// camera, `y`, as the near cascades see what lies nearer.
const FAR_SHADOW_LIFT = vec2<f32>(8.0, 200.0);
// Texels along a side of each cascade, as in `shadows.rs`.
const SHADOW_MAP_SIZE = 2048.0;

// How much of the light reaches `position` in `cascade`, counting only what
// lies more than `lift` blocks towards the light. The shadow's edge is as
// soft as the sun's disc makes it: sharp where what casts it is near, softer
// the farther it is.
fn cascade_shadow(cascade: i32, position: vec3<f32>, normal: vec3<f32>, facing: f32, lift: f32) -> f32 {
    // Nudged out along the normal by about a texel, more where the light
    // grazes the face, so faces don't shade themselves.
    let texel = 2.0 * globals.shadow[cascade] / SHADOW_MAP_SIZE;
    let lifted = position + normal * texel * (1.0 + 2.0 * (1.0 - facing)) + globals.light.xyz * lift;
    let matrix = globals.shadow_view_proj[cascade];
    let light = matrix * vec4<f32>(lifted, 1.0);
    let uv = vec2<f32>(light.x * 0.5 + 0.5, 0.5 - light.y * 0.5);
    if any(uv <= vec2<f32>(0.0)) || any(uv >= vec2<f32>(1.0)) || light.z >= 1.0 {
        return 1.0;
    }
    let layer = i32(globals.shadow_layers[cascade]);
    // How much of the map a block is.
    let across = 0.5 / globals.shadow[cascade];
    // How far what casts the shadow is: the blockers over the softest edge
    // worth looking for here, on average.
    let search = max(clamp(SUN_WIDTH * globals.shadow[cascade], 0.2, 8.0) * across, 2.0 / SHADOW_MAP_SIZE);
    if cascade == 3 && nothing_nearer(uv, light.z, search, facing) {
        return 1.0;
    }
    let slope = depth_slope(matrix, normal);
    // Depth per block towards the light.
    let per_block = abs((matrix * vec4<f32>(globals.light.xyz, 0.0)).z);
    var blockers = 0.0;
    var gap = 0.0;
    var around = vec2<f32>(search, 0.0);
    let quarter = mat2x2<f32>(0.0, 1.0, -1.0, 0.0);
    for (var i = 0; i < 5; i++) {
        var offset = vec2<f32>(0.0);
        if i > 0 {
            offset = around;
            around = quarter * around;
        }
        let texel_at = clamp(
            vec2<i32>((uv + offset) * SHADOW_MAP_SIZE),
            vec2<i32>(0),
            vec2<i32>(i32(SHADOW_MAP_SIZE) - 1),
        );
        let stored = textureLoad(shadow_depths, texel_at, layer, 0).r;
        let here = light.z + dot(slope, offset);
        if stored < here - 1e-5 {
            blockers += 1.0;
            gap += here - stored;
        }
    }
    if blockers == 0.0 {
        return 1.0;
    }
    let distance = gap / blockers / per_block + lift;
    let width = SHADOW_SOFTNESS + SUN_WIDTH * distance;
    // No sharper than a couple of texels.
    let radius = max(0.5 * width * across, 1.5 / SHADOW_MAP_SIZE);
    return soft_shadow(uv, light.z, slope, radius, layer);
}


// Whether the far cascade surely holds nothing that would count as a
// blocker below: nothing nearer the light, around `uv`, than `depth` and
// as much nearer as a face turned `facing` towards the light gets across
// `search`. One read instead of the search, for most points.
fn nothing_nearer(uv: vec2<f32>, depth: f32, search: f32, facing: f32) -> bool {
    // The hint covers the squares around, so the search mustn't reach
    // farther.
    if search * SHADOW_MAP_SIZE >= f32(SHADOW_HINT_BLOCK) - 1.0 {
        return false;
    }
    let texel = clamp(
        vec2<i32>(uv * f32(SHADOW_HINT_SIZE)),
        vec2<i32>(0),
        vec2<i32>(SHADOW_HINT_SIZE - 1),
    );
    let nearest = bitcast<f32>(textureLoad(shadow_hint, texel, 0).r);
    // How much depth a face gains per unit across the map at most: the
    // tangent of its angle to the light, in the cascade's units, no more
    // than `depth_slope` allows.
    let radius = globals.shadow[3];
    let tangent = sqrt(max(1.0 - facing * facing, 0.0)) / max(facing, 1e-4);
    let slope = min(tangent * radius / (radius + FAR_CASTER_MARGIN) * 1.01, 1.0);
    return nearest >= depth + slope * search;
}

// Builds `shadow_hint` from a layer of the shadow map, in two passes over
// one triangle covering it: the nearest depth in each square, then of each
// square and those around it.
@group(0) @binding(20) var hint_source: texture_2d<f32>;
@group(0) @binding(21) var hint_blocks: texture_2d<u32>;

@vertex
fn hint_vertex(@builtin(vertex_index) index: u32) -> @builtin(position) vec4<f32> {
    return vec4<f32>(f32(index == 1u) * 4.0 - 1.0, f32(index == 2u) * 4.0 - 1.0, 0.0, 1.0);
}

@fragment
fn hint_blocks_fragment(@builtin(position) at: vec4<f32>) -> @location(0) u32 {
    let first = vec2<i32>(at.xy) * SHADOW_HINT_BLOCK;
    var nearest = 1.0;
    for (var y = 0; y < SHADOW_HINT_BLOCK; y++) {
        for (var x = 0; x < SHADOW_HINT_BLOCK; x++) {
            nearest = min(nearest, textureLoad(hint_source, first + vec2<i32>(x, y), 0).r);
        }
    }
    return bitcast<u32>(nearest);
}

@fragment
fn hint_fragment(@builtin(position) at: vec4<f32>) -> @location(0) u32 {
    let texel = vec2<i32>(at.xy);
    var nearest = 1.0;
    for (var y = -1; y <= 1; y++) {
        for (var x = -1; x <= 1; x++) {
            let near = clamp(texel + vec2<i32>(x, y), vec2<i32>(0), vec2<i32>(SHADOW_HINT_SIZE - 1));
            nearest = min(nearest, bitcast<f32>(textureLoad(hint_blocks, near, 0).r));
        }
    }
    return bitcast<u32>(nearest);
}

// How soft the edge of a shadow is at the least, in blocks: the air
// scatters a little light around everything.
const SHADOW_SOFTNESS = 0.04;
// Comparisons averaged for a soft edge: each texel of the shadow map
// counts for little, so edges don't twitch as the sun moves across the
// texels.
const SHADOW_TAPS = 32;

// The share of comparisons over a disc of `radius` that find no shadow.
// The disc is filled in a sunflower pattern, each comparison bilinearly
// filtered. Each is made with the depth the face would have there rather
// than at the middle, so a wide disc doesn't shade the face it lies on.
fn soft_shadow(uv: vec2<f32>, depth: f32, slope: vec2<f32>, radius: f32, layer: i32) -> f32 {
    // Most points are in full light or full shadow: if the middle and a
    // ring around it agree, so does the rest of the disc.
    var quick = shadow_tap(uv, depth, slope, vec2<f32>(0.0), layer);
    let eighth = mat2x2<f32>(0.7071068, 0.7071068, -0.7071068, 0.7071068);
    var around = vec2<f32>(0.9 * radius, 0.0);
    for (var i = 0; i < 8; i++) {
        quick += shadow_tap(uv, depth, slope, around, layer);
        around = eighth * around;
    }
    if quick == 0.0 || quick == 9.0 {
        return quick / 9.0;
    }
    // The golden angle.
    let turn = mat2x2<f32>(-0.7373688, 0.6754903, -0.6754903, -0.7373688);
    var direction = vec2<f32>(1.0, 0.0);
    var sum = 0.0;
    for (var i = 0; i < SHADOW_TAPS; i++) {
        let offset = direction * sqrt((f32(i) + 0.5) / f32(SHADOW_TAPS)) * radius;
        sum += shadow_tap(uv, depth, slope, offset, layer);
        direction = turn * direction;
    }
    return sum / f32(SHADOW_TAPS);
}

// One filtered comparison, `offset` across the shadow map from `uv`.
fn shadow_tap(uv: vec2<f32>, depth: f32, slope: vec2<f32>, offset: vec2<f32>, layer: i32) -> f32 {
    return textureSampleCompareLevel(
        shadow_map,
        shadow_sampler,
        uv + offset,
        layer,
        depth + dot(slope, offset),
    );
}

// How depth from the light changes across the shadow map along a face
// turned towards `normal`: per unit of u and of v.
fn depth_slope(matrix: mat4x4<f32>, normal: vec3<f32>) -> vec2<f32> {
    let side = select(vec3<f32>(0.0, 1.0, 0.0), vec3<f32>(1.0, 0.0, 0.0), abs(normal.y) > 0.5);
    let along = normalize(cross(normal, side));
    let across = cross(normal, along);
    let l1 = (matrix * vec4<f32>(along, 0.0)).xyz;
    let l2 = (matrix * vec4<f32>(across, 0.0)).xyz;
    // How far across the map a block along the face goes: u to the right,
    // v down.
    let a = vec2<f32>(0.5 * l1.x, -0.5 * l1.y);
    let b = vec2<f32>(0.5 * l2.x, -0.5 * l2.y);
    let det = a.x * b.y - a.y * b.x;
    if abs(det) < 1e-12 {
        return vec2<f32>(0.0);
    }
    let slope = vec2<f32>(l1.z * b.y - a.y * l2.z, a.x * l2.z - l1.z * b.x) / det;
    // Where the light grazes the face the slope grows without bound; the
    // face gets little light there anyway.
    let steepness = length(slope);
    return slope * min(1.0, 1.0 / max(steepness, 1e-6));
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
    let face = (quad.x >> 24u) & 7u;
    let corner = lod_corner(quad, vertex & 3u);
    let local = corner.local;
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

// Far-away terrain drawn into the far cascade of the shadow map.
@vertex
fn lod_shadow_vertex(@builtin(vertex_index) vertex: u32, @location(0) quad: vec4<u32>) -> @builtin(position) vec4<f32> {
    let local = lod_corner(quad, vertex & 3u).local;
    return shadow_pass * vec4<f32>(camera_relative(quad.y >> 18u, local), 1.0);
}

// A corner of a packed quad of far-away terrain (see `lod.rs`), in its tile.
fn lod_corner(quad: vec4<u32>, vertex: u32) -> BoxCorner {
    let first = vec3<f32>(f32(quad.x & 63u), 0.0, f32((quad.x >> 6u) & 63u)) * 4.0;
    let cells = vec3<f32>(f32(((quad.x >> 12u) & 63u) + 1u), 0.0, f32(((quad.x >> 18u) & 63u) + 1u));
    let face = (quad.x >> 24u) & 7u;
    // The box's bottom and top, sign-extended from 16 bits.
    let low = f32(bitcast<i32>(quad.z << 16u) >> 16u);
    let high = f32(bitcast<i32>(quad.z) >> 16u);
    let start = vec3<f32>(first.x, low, first.z);
    let size = vec3<f32>(cells.x * 4.0, high - low, cells.z * 4.0);
    var corner = box_corner(start, size, face, vertex);
    // A little below the chunks drawn in full, so where both are drawn,
    // those win.
    corner.local.y -= 0.5;
    return corner;
}

@fragment
fn lod_fragment(in: ChunkVertex) -> @location(0) vec4<f32> {
    // Where chunks are drawn in full, they take over; until they are, far
    // terrain stands in for them. A face belongs to the block behind it.
    let inside = floor(in.position + globals.camera_fract.xyz - in.normal * 0.5);
    if chunk_drawn(globals.camera_block.xyz + vec3<i32>(inside)) {
        discard;
    }
    return shade(in);
}

// Whether the chunk holding `block` is drawn in full this frame.
fn chunk_drawn(block: vec3<i32>) -> bool {
    let chunk = block >> vec3<u32>(5u);
    // The map repeats: it only holds the columns around the camera.
    let camera = globals.camera_block.xz >> vec2<u32>(5u);
    if any(abs(chunk.xz - camera) >= vec2<i32>(DRAWN_WIDTH / 2)) {
        return false;
    }
    let bits = textureLoad(chunks_drawn, chunk.xz & vec2<i32>(DRAWN_WIDTH - 1), 0).r;
    return ((bits >> u32(chunk.y & 31)) & 1u) != 0u;
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
    // Without a map of shadows, the air by a face out of the sky's sight is
    // taken to be in the shade; with the light low, when shadows grow long,
    // so is the air by a face turned away from it.
    let low = 1.0 - smoothstep(0.1, 0.35, globals.light.y);
    let facing_light = smoothstep(0.7, 1.0, in.light.x)
        * mix(1.0, smoothstep(-0.2, 0.1, dot(in.normal, globals.light.xyz)), low);
    return vec4<f32>(finish(with_fog(color, in.position, distance, in.clip.xy, facing_light)), 1.0);
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
    if globals.fog.z > 0.0 {
        // Where the sky is out of sight, as in caves, so is the light:
        // nothing to trace.
        let open = smoothstep(0.3, 0.6, sky);
        var shadow = 0.0;
        if open > 0.0 && globals.light.w > 0.0 {
            shadow = sunlit(position, normal, distance, facing) * open;
        }
        // As the light sinks to the horizon its shadows fade out, and the
        // light goes with them: it never reaches what they hid, and the
        // valleys don't flash red at sunset.
        visible = shadow * globals.light.w;
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

// Distant things fade into the haze, and at the edge of the world into what
// lies behind them. Where hills keep the light off the air in between, it
// only scatters the sky's light: a mountain in front of the setting sun
// stays dark instead of showing the glow around the sun. `pixel` is where
// on the screen, to vary the samples of the air; `guess` how much of the
// air is lit where no shadow map tells.
fn with_fog(color: vec3<f32>, position: vec3<f32>, distance: f32, pixel: vec2<f32>, guess: f32) -> vec3<f32> {
    let direction = position / max(distance, 1e-4);
    let amount = haze(position, distance);
    let lit_haze = haze_radiance(direction);
    let shaded_haze = max(globals.ambient[0].rgb, vec3<f32>(0.0)) * globals.weather.y;
    var air = 1.0;
    // Only where the difference would show.
    let glow = amount * dot(max(lit_haze - shaded_haze, vec3<f32>(0.0)), LUMINANCE);
    if glow * globals.screen.z > AIR_SHADE_VISIBLE {
        air = air_in_light(position, distance, pixel, guess);
    }
    let hazy = mix(color, mix(shaded_haze, lit_haze, air), amount);
    let edge = smoothstep(mix(globals.fog.x, globals.fog.y, 0.5), globals.fog.y, distance);
    // Nearer than the edge, as nearly everything is, the sky behind doesn't
    // show at all.
    if edge <= 0.0 {
        return hazy;
    }
    return mix(hazy, background(direction), edge);
}

// Below this, in light after exposure, the shade on the haze wouldn't show.
const AIR_SHADE_VISIBLE = 0.002;
// Points along a view ray checked for light.
const AIR_SAMPLES = 8;

// How much of the haze between the camera and `position` sees the light,
// from 0 to 1, by points along the way, each counted by how much haze is
// there; from the far cascade of the shadow map, which holds the hills, or
// from the widest near one without it. With no map, `guess`.
fn air_in_light(position: vec3<f32>, distance: f32, pixel: vec2<f32>, guess: f32) -> f32 {
    // Sinking to the horizon, the sun leaves the air by the ground as it
    // leaves the ground, as fast as its shadows fade out (`strength` in
    // `world_pass.rs`); the glow it leaves in the sky isn't the air's here.
    let up = smoothstep(0.0, 0.02, globals.sun.y);
    let reach = globals.shadow;
    var cascade = 3;
    if reach.w <= reach.z {
        // Rays take the place of the near cascades.
        if globals.ray_window.w > 0 {
            return guess * up;
        }
        cascade = 2;
    }
    if globals.light.w <= 0.0 {
        return guess * up;
    }
    let matrix = globals.shadow_view_proj[cascade];
    let layer = i32(globals.shadow_layers[cascade]);
    let radius = reach[cascade];
    let margin = select(CASTER_MARGIN, FAR_CASTER_MARGIN, cascade == 3);
    // A block towards the light, in depth: air right by a lit face is lit.
    let lift = 1.0 / (2.0 * (radius + margin));
    let along = min(distance, radius) / max(distance, 1e-4);
    // Where along the way differs from pixel to pixel, so the few points
    // don't show as bands (interleaved gradient noise, after Jimenez).
    let jitter = fract(52.9829189 * fract(dot(pixel, vec2<f32>(0.06711056, 0.00583715))));
    let eye = max(camera_height(), 0.0);
    var lit = 0.0;
    var total = 0.0;
    for (var i = 0; i < AIR_SAMPLES; i++) {
        let point = position * ((f32(i) + jitter) / f32(AIR_SAMPLES) * along);
        let density = exp(-max(eye + point.y, 0.0) / HAZE_HEIGHT);
        let light = matrix * vec4<f32>(point, 1.0);
        let uv = vec2<f32>(light.x * 0.5 + 0.5, 0.5 - light.y * 0.5);
        var seen = 1.0;
        if all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0)) && light.z < 1.0 {
            seen = textureSampleCompareLevel(shadow_map, shadow_sampler, uv, layer, light.z - lift);
        }
        lit += seen * density;
        total += density;
    }
    return lit / total * up;
}

// What the sky shows in `direction`: below the horizon, haze over land too
// far to draw.
fn background(direction: vec3<f32>) -> vec3<f32> {
    let level = normalize(vec3<f32>(direction.x, max(direction.y, 0.0), direction.z));
    let sky = sky_radiance(level);
    if direction.y >= 0.0 {
        return sky;
    }
    return mix(sky, haze_radiance(direction), smoothstep(0.0, 0.05, -direction.y));
}

// The haze is lit from this many radians up the sky: as bright as the sky
// there, the same angle from the light. The bright band at the horizon
// comes from looking through all of the air, while haze lies a few
// kilometres deep at most.
const HAZE_ELEVATION = 0.1;

// The light the haze scatters towards the eye in `direction`: brighter
// towards the sun, whatever the height.
fn haze_radiance(direction: vec3<f32>) -> vec3<f32> {
    var color = vec3<f32>(0.0);
    if globals.sky.x >= 0.0 {
        color += haze_from(direction, globals.sun.xyz, globals.sky.x);
    }
    if globals.sky.y >= 0.0 {
        color += haze_from(direction, -globals.sun.xyz, globals.sky.y) * globals.sky.z;
    }
    let grey = dot(color, LUMINANCE);
    return mix(color, vec3<f32>(grey), globals.weather.x * 0.5) * globals.weather.y;
}

// The sky at `HAZE_ELEVATION`, as far from the light `toward` as `direction`
// is.
fn haze_from(direction: vec3<f32>, toward: vec3<f32>, layer: f32) -> vec3<f32> {
    let up = vec2<f32>(sin(HAZE_ELEVATION), cos(HAZE_ELEVATION));
    let level = max(length(toward.xz), 1e-4);
    let around = acos(clamp((dot(direction, toward) - up.x * toward.y) / (up.y * level), -1.0, 1.0));
    let bearing = atan2(toward.z, toward.x) + around;
    let sample = vec3<f32>(up.y * cos(bearing), up.x, up.y * sin(bearing));
    return sky_from(sample, toward, layer);
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
    var color: vec3<f32>;
    if classic() {
        // Below the horizon, where nothing is drawn, the horizon goes on.
        let level = normalize(vec3<f32>(direction.x, max(direction.y, 0.0), direction.z));
        color = output(classic_sky(level));
    } else {
        color = finish(background(direction));
    }

    // Stars: one in a few hundred cells of a grid around the camera. They
    // are added on screen, whatever the exposure.
    let up = direction.y;
    if globals.sky.w > 0.0 && up > 0.0 && !behind_disc(direction) {
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

// Whether the sun's or the moon's disc hides the sky in `direction`: no
// stars show through them.
fn behind_disc(direction: vec3<f32>) -> bool {
    for (var body = 0u; body < 2u; body++) {
        let center = select(globals.sun.xyz, -globals.sun.xyz, body == 1u);
        let along = dot(direction, center);
        if along <= 0.0 {
            continue;
        }
        let right = normalize(cross(vec3<f32>(0.0, 0.0, 1.0), center));
        let up = cross(center, right);
        // Where on the disc's square, as in `celestial_vertex`.
        let corner = vec2<f32>(dot(direction, right), dot(direction, up)) / along / disc_size(body);
        if all(abs(corner) < vec2<f32>(1.0)) {
            let uv = vec2<f32>(corner.x, -corner.y) * 0.5 + 0.5;
            if textureSampleLevel(sky_textures, block_sampler, uv, body, 0.0).a > 0.5 {
                return true;
            }
        }
    }
    return false;
}

// Half the width of the square of the sun (`body` 0) or the moon (1), as a
// tangent of the angle.
fn disc_size(body: u32) -> f32 {
    return select(0.12, 0.09, body == 1u);
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
    let size = disc_size(body);
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

// Clouds: blocks of cloud in a thin layer, as in classic block games. A
// cloud isn't geometry: each pixel walks its ray across the cells of the
// layer until it meets one. See `clouds.rs`.

// The cell map and the obstacle map, as in `clouds.rs`.
const CLOUD_CELL = 12.0;
const CELL_MAP_SIZE = 1024;
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
// How much light clouds reflect.
const CLOUD_ALBEDO = 0.9;
// How much a block of cloud dims the light passing through it, for
// fair-weather clouds (density 1): looking straight up through the layer,
// the sky shows through a fifth. The same for the eye as for the sun, so a
// cloud the sun shines through casts as light a shadow.
const CLOUD_EXTINCTION = 0.27;
// Clouds scatter light mostly onwards: some of what they take from the
// sun's beam still reaches the ground under them, as if through a cloud
// thinner by `y`, weighted `x` against the beam.
const CLOUD_ONWARD = vec2<f32>(0.3, 0.25);
// The sun is a disc this many radians across: a shadow's edge is this much
// softer per block from what casts it.
const SUN_WIDTH = 0.0093;
// Light scattered about inside a cloud softens its shadow's edges by this
// many blocks besides, and onwards by this much more per block it comes
// from the cloud.
const CLOUD_SHADOW_SOFT = 1.0;
const CLOUD_SHADOW_SPREAD = 0.01;
// The farther below its cloud, the more of a shadow light from all around
// fills in: half of it over this many blocks or so.
const CLOUD_SHADOW_FILL = 1000.0;
// Cloud shadows fade out from this far away to that, in the haze.
const CLOUD_SHADOW_FADE = vec2<f32>(250.0, 900.0);

fn camera_height() -> f32 {
    return f32(globals.camera_block.y) + globals.camera_fract.y;
}

// How high obstacles near `xz` (relative to the camera) reach, as clouds
// keep clear of them.
fn obstacle_top(xz: vec2<f32>) -> f32 {
    let uv = (xz - globals.cloud_place.zw) / OBSTACLE_SPAN;
    let inside = all(uv > vec2<f32>(0.0)) && all(uv < vec2<f32>(1.0));
    let height = textureSampleLevel(cloud_obstacles, smooth_sampler, uv, 0.0).r * 255.0;
    return obstacle_base() + select(0.0, height, inside);
}

// Whether `cell` of the layer holds cloud. Cells are counted over the air
// the wind carries.
fn cloud_cell(cell: vec2<i32>) -> bool {
    let texel = (cell % CELL_MAP_SIZE + CELL_MAP_SIZE) % CELL_MAP_SIZE;
    return textureLoad(cloud_cells, texel, 0).r > 0.5;
}

// How much of a box `size` blocks across around `xz` (relative to the camera)
// is cloud, from 0 to 1. The box slides over cell edges smoothly, so this
// moves smoothly with the clouds.
//
// Filtering the cell map between texels does it in one sample: across a
// cell, the sample stays on the cell's texel but within half the box of its
// sides, where it moves on to the next linearly, as the box sliding over the
// edge would. The box can be no wider than a cell.
//
// Wider than a cell, by `level` halvings, the smaller levels of the map
// blur it instead.
fn cloud_cover(xz: vec2<f32>, size: vec2<f32>, level: f32) -> f32 {
    if level > 0.0 {
        let uv = (globals.cloud_place.xy + xz) / CLOUD_CELL / f32(CELL_MAP_SIZE);
        return textureSampleLevel(cloud_cells, wrap_sampler, uv, level).r;
    }
    let at = (globals.cloud_place.xy + xz) / CLOUD_CELL - 0.5;
    let first = floor(at);
    let f = clamp((at - first - 0.5) * (CLOUD_CELL / size) + 0.5, vec2<f32>(0.0), vec2<f32>(1.0));
    let uv = (first + 0.5 + f) / f32(CELL_MAP_SIZE);
    return textureSampleLevel(cloud_cells, wrap_sampler, uv, 0.0).r;
}

// How much of the light gets through the clouds to `position`, `distance`
// from the eye, on its way along `toward`. The light dims with how much
// cloud it crosses (Beer's law), and some of what the cloud scatters onwards
// still arrives. The farther below the cloud, the softer the shadow's
// edges, as the sun's disc and the scattered light make them, and the more
// light from all around fills it in. Worked out for each point from the
// cells of cloud the light crosses, so shadows move exactly as the clouds
// do.
fn cloud_shadow(position: vec3<f32>, toward: vec3<f32>, distance: f32) -> f32 {
    let height = position.y + camera_height();
    let density = globals.clouds.y;
    if density == 0.0 || toward.y < 0.02 || height >= cloud_top() {
        return 1.0;
    }
    // Far away the haze washes cloud shadows out; with the sun low, light
    // reaches under the clouds from the side.
    let fade = (1.0 - smoothstep(CLOUD_SHADOW_FADE.x, CLOUD_SHADOW_FADE.y, distance))
        * smoothstep(0.02, 0.1, toward.y);
    if fade <= 0.0 {
        return 1.0;
    }
    // Where the light on its way here comes into the layer of clouds, how
    // far it goes through it, and how far that is across.
    let start = max(cloud_bottom(), height);
    let entry = position.xz + toward.xz * ((start - height) / toward.y);
    let way = (cloud_top() - start) / toward.y;
    let across = toward.xz * way;
    // How soft the shadow's edges are, from how far below the middle of the
    // layer.
    let below = max(0.5 * (cloud_bottom() + cloud_top()) - height, 0.0) / toward.y;
    let soft = CLOUD_SHADOW_SOFT + (SUN_WIDTH + CLOUD_SHADOW_SPREAD) * below;
    let level = log2(max(soft / CLOUD_CELL, 1.0));
    // How much of the way is in cloud: boxes along it, each reaching the
    // next, so together they cover it all.
    let steps = clamp(ceil(max(abs(across.x), abs(across.y)) / CLOUD_CELL), 1.0, 8.0);
    let size = clamp(max(vec2<f32>(soft), abs(across) / steps), vec2<f32>(0.5), vec2<f32>(CLOUD_CELL));
    var cloud = 0.0;
    for (var i = 0; i < 8; i++) {
        if f32(i) >= steps {
            break;
        }
        cloud += cloud_cover(entry + across * ((f32(i) + 0.5) / steps), size, level);
    }
    if cloud <= 0.0 {
        return 1.0;
    }
    // Thinning out near the tops of obstacles, as the clouds do.
    let obstacle = obstacle_top(entry + across * 0.5);
    let clear = smoothstep(obstacle + 1.0, obstacle + 8.0, 0.5 * (start + cloud_top()));
    let depth = CLOUD_EXTINCTION * density * clear * way * cloud / steps;
    let through = (exp(-depth) + CLOUD_ONWARD.x * exp(-depth * CLOUD_ONWARD.y))
        / (1.0 + CLOUD_ONWARD.x);
    let filled = 1.0 - exp(-below / CLOUD_SHADOW_FILL);
    return mix(1.0, mix(through, 1.0, filled), fade);
}

fn view_direction(screen: vec2<f32>) -> vec3<f32> {
    let far = globals.inverse_view_proj * vec4<f32>(screen, 1.0, 1.0);
    return normalize(far.xyz / far.w);
}

// Clouds drawn over the screen, and how far away they are.
struct ScreenClouds {
    @location(0) color: vec4<f32>,
    // So the world in front of them hides them.
    @builtin(frag_depth) depth: f32,
}

// The clouds, drawn over the screen: each pixel walks its ray across the
// cells of the layer until it meets a cloud.
@fragment
fn cloud_fragment(in: SkyVertex) -> ScreenClouds {
    let direction = view_direction(in.screen);
    let bottom = cloud_bottom() - camera_height();
    let top = cloud_top() - camera_height();
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
    let begin = globals.cloud_place.xy + flat * start;
    var cell = vec2<i32>(floor(begin / CLOUD_CELL));
    let step = vec2<i32>(select(vec2<f32>(-1.0), vec2<f32>(1.0), flat >= vec2<f32>(0.0)));
    let across = CLOUD_CELL / max(abs(flat), vec2<f32>(1e-6));
    let first_edge = (vec2<f32>(cell) + select(vec2<f32>(0.0), vec2<f32>(1.0), flat >= vec2<f32>(0.0)))
        * CLOUD_CELL;
    var next = start + (first_edge - begin) / select(flat, vec2<f32>(1e-6), abs(flat) < vec2<f32>(1e-6));
    next = select(next, vec2<f32>(1e9), abs(flat) < vec2<f32>(1e-6));
    var t = start;
    // Entered through the bottom or the top, or a side along x or z.
    var side = 0;
    var hit = false;
    for (var i = 0; i < 160; i++) {
        if cloud_cell(cell) {
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
    // How much cloud the eye looks through: on through the layer, cell by
    // cell, until little shows through.
    var inside = 0.0;
    var at = t;
    var walk = cell;
    var edge = next;
    for (var i = 0; i < 64; i++) {
        let leave = min(min(edge.x, edge.y), end);
        if cloud_cell(walk) {
            inside += leave - at;
        }
        if leave >= end || inside * CLOUD_EXTINCTION * globals.clouds.y > 5.0 {
            break;
        }
        at = leave;
        if edge.x < edge.y {
            edge.x += across.x;
            walk.x += step.x;
        } else {
            edge.y += across.y;
            walk.y += step.y;
        }
    }
    let position = direction * max(t, 0.05);
    // Never inside buildings and mountains: thinning out near their tops.
    let obstacle = obstacle_top(position.xz);
    let clear = smoothstep(obstacle + 1.0, obstacle + 8.0, position.y + camera_height());
    let fade = (1.0 - smoothstep(reach * 0.6, reach, t)) * clear;
    if fade <= 0.0 {
        discard;
    }
    // Denser clouds are greyer and let less of the sky through; at density
    // 1, as in classic block games.
    let density = globals.clouds.y;
    let grey = 1.0 / (1.0 + 0.4 * max(density - 1.0, 0.0));
    var color: vec3<f32>;
    if classic() {
        // Lit from above, darker underneath.
        var shade = select(0.86, 1.0, normal.y > 0.5);
        shade = select(shade, 0.7, normal.y < -0.5);
        let fog = smoothstep(globals.fog.x, globals.fog.y, t);
        color = output(mix(globals.classic[5].rgb * shade * grey, globals.classic[3].rgb, fog));
    } else {
        let facing = dot(normal, globals.cloud_light.xyz) * 0.75 + 0.25;
        let lit = globals.cloud_light_color.rgb * max(facing, 0.0) + sky_light(normal);
        color = finish(mix(CLOUD_ALBEDO * grey * lit, sky_radiance(direction), haze(position, t)));
    }
    // As much of what's behind shows through as of the sun's light gets
    // through the same cloud; in classic light, as in classic block games.
    var cover = 1.0 - exp(-CLOUD_EXTINCTION * density * inside);
    if classic() {
        cover = 1.0 - pow(0.2, density);
    }
    var out: ScreenClouds;
    out.color = vec4<f32>(color, cover * fade);
    let clip = globals.view_proj * vec4<f32>(position, 1.0);
    out.depth = clamp(clip.z / clip.w, 0.0, 1.0);
    return out;
}
