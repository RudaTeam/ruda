// After the world is drawn, in light that can be far brighter than a screen
// shows: bright light bleeds into its surroundings (bloom), the exposure
// follows how bright the view is, and the result is mapped to the screen.
//
// Bloom follows "Next Generation Post Processing in Call of Duty: Advanced
// Warfare" (Jimenez, 2014): the frame is halved again and again with a
// 13-tap filter, then each level is blurred back up into the one above it.

struct Post {
    // x: seconds since the last frame; y: 1 to set the exposure at once;
    // z: how much of the bloom shows; w: 1 if the shader must encode sRGB
    // itself.
    frame: vec4<f32>,
    // x, y: least and most exposure; z: the brightness a view of average
    // brightness is shown at; w: how far the exposure follows the view's
    // brightness, 0 not at all, 1 fully.
    exposure: vec4<f32>,
    // x: 1 to draw the crosshair; y, z: screen size in pixels.
    overlay: vec4<f32>,
}

@group(0) @binding(0) var<uniform> post: Post;
@group(0) @binding(1) var linear_sampler: sampler;
// What a pass reads: the frame, or a level of bloom.
@group(0) @binding(2) var source: texture_2d<f32>;
// The finished bloom, for the last pass.
@group(0) @binding(3) var bloom: texture_2d<f32>;
// One texel: the exposure, last frame's while adapting, this frame's after.
@group(0) @binding(4) var exposure: texture_2d<f32>;

struct Fullscreen {
    @builtin(position) clip: vec4<f32>,
    @location(0) uv: vec2<f32>,
}

// One triangle covering the screen.
@vertex
fn fullscreen(@builtin(vertex_index) index: u32) -> Fullscreen {
    let uv = vec2<f32>(f32((index << 1u) & 2u), f32(index & 2u));
    var out: Fullscreen;
    out.clip = vec4<f32>(uv.x * 2.0 - 1.0, 1.0 - uv.y * 2.0, 0.0, 1.0);
    out.uv = uv;
    return out;
}

fn luminance(color: vec3<f32>) -> f32 {
    return dot(color, vec3<f32>(0.2126, 0.7152, 0.0722));
}

fn tap(uv: vec2<f32>, offset: vec2<f32>) -> vec3<f32> {
    let texel = 1.0 / vec2<f32>(textureDimensions(source));
    // Never more than half-precision floats hold, nor less than nothing.
    return clamp(textureSampleLevel(source, linear_sampler, uv + offset * texel, 0.0).rgb, vec3<f32>(0.0), vec3<f32>(60000.0));
}

// Four taps averaged, each weighted down the brighter it is, so a single
// very bright pixel doesn't flicker as a big blob of bloom.
fn tame(a: vec3<f32>, b: vec3<f32>, c: vec3<f32>, d: vec3<f32>) -> vec3<f32> {
    let wa = 1.0 / (1.0 + luminance(a));
    let wb = 1.0 / (1.0 + luminance(b));
    let wc = 1.0 / (1.0 + luminance(c));
    let wd = 1.0 / (1.0 + luminance(d));
    return (a * wa + b * wb + c * wc + d * wd) / (wa + wb + wc + wd);
}

// Thirteen taps around a texel of the half-size level being made.
@fragment
fn downsample_first(in: Fullscreen) -> @location(0) vec4<f32> {
    let a = tap(in.uv, vec2<f32>(-2.0, -2.0));
    let b = tap(in.uv, vec2<f32>(0.0, -2.0));
    let c = tap(in.uv, vec2<f32>(2.0, -2.0));
    let d = tap(in.uv, vec2<f32>(-2.0, 0.0));
    let e = tap(in.uv, vec2<f32>(0.0, 0.0));
    let f = tap(in.uv, vec2<f32>(2.0, 0.0));
    let g = tap(in.uv, vec2<f32>(-2.0, 2.0));
    let h = tap(in.uv, vec2<f32>(0.0, 2.0));
    let i = tap(in.uv, vec2<f32>(2.0, 2.0));
    let j = tap(in.uv, vec2<f32>(-1.0, -1.0));
    let k = tap(in.uv, vec2<f32>(1.0, -1.0));
    let l = tap(in.uv, vec2<f32>(-1.0, 1.0));
    let m = tap(in.uv, vec2<f32>(1.0, 1.0));
    let color = tame(j, k, l, m) * 0.5
        + (tame(a, b, d, e) + tame(b, c, e, f) + tame(d, e, g, h) + tame(e, f, h, i)) * 0.125;
    return vec4<f32>(color, 1.0);
}

@fragment
fn downsample(in: Fullscreen) -> @location(0) vec4<f32> {
    let a = tap(in.uv, vec2<f32>(-2.0, -2.0));
    let b = tap(in.uv, vec2<f32>(0.0, -2.0));
    let c = tap(in.uv, vec2<f32>(2.0, -2.0));
    let d = tap(in.uv, vec2<f32>(-2.0, 0.0));
    let e = tap(in.uv, vec2<f32>(0.0, 0.0));
    let f = tap(in.uv, vec2<f32>(2.0, 0.0));
    let g = tap(in.uv, vec2<f32>(-2.0, 2.0));
    let h = tap(in.uv, vec2<f32>(0.0, 2.0));
    let i = tap(in.uv, vec2<f32>(2.0, 2.0));
    let j = tap(in.uv, vec2<f32>(-1.0, -1.0));
    let k = tap(in.uv, vec2<f32>(1.0, -1.0));
    let l = tap(in.uv, vec2<f32>(-1.0, 1.0));
    let m = tap(in.uv, vec2<f32>(1.0, 1.0));
    let color = e * 0.125
        + (a + c + g + i) * 0.03125
        + (b + d + f + h) * 0.0625
        + (j + k + l + m) * 0.125;
    return vec4<f32>(color, 1.0);
}

// A level blurred with a 3×3 tent, added onto the level above it.
@fragment
fn upsample(in: Fullscreen) -> @location(0) vec4<f32> {
    var color = tap(in.uv, vec2<f32>(0.0, 0.0)) * 4.0;
    color += (tap(in.uv, vec2<f32>(-1.0, 0.0)) + tap(in.uv, vec2<f32>(1.0, 0.0))
        + tap(in.uv, vec2<f32>(0.0, -1.0)) + tap(in.uv, vec2<f32>(0.0, 1.0))) * 2.0;
    color += tap(in.uv, vec2<f32>(-1.0, -1.0)) + tap(in.uv, vec2<f32>(1.0, -1.0))
        + tap(in.uv, vec2<f32>(-1.0, 1.0)) + tap(in.uv, vec2<f32>(1.0, 1.0));
    return vec4<f32>(color / 16.0, 1.0);
}

// The exposure for this frame, from the smallest level of bloom: how bright
// the view is on average, the middle counting most. Like eyes, it adapts
// to the dark slowly and to bright light quickly.
@fragment
fn adapt(in: Fullscreen) -> @location(0) vec4<f32> {
    var total = 0.0;
    var weights = 0.0;
    for (var y = 0u; y < 5u; y++) {
        for (var x = 0u; x < 5u; x++) {
            let uv = (vec2<f32>(f32(x), f32(y)) + 0.5) / 5.0;
            let color = textureSampleLevel(source, linear_sampler, uv, 0.0).rgb;
            let weight = 1.5 - length(uv - 0.5) * 1.4;
            total += log2(max(luminance(color), 1e-4)) * weight;
            weights += weight;
        }
    }
    let average = exp2(total / weights);
    let wanted = clamp(
        post.exposure.z / pow(average, post.exposure.w),
        post.exposure.x,
        post.exposure.y,
    );
    let previous = textureLoad(exposure, vec2<i32>(0), 0).r;
    if post.frame.y > 0.5 || !(previous > 0.0 && previous < 1e4) {
        return vec4<f32>(wanted, 0.0, 0.0, 1.0);
    }
    let rate = select(1.2, 2.5, wanted < previous);
    let step = 1.0 - exp(-post.frame.x * rate);
    return vec4<f32>(exp2(mix(log2(previous), log2(wanted), step)), 0.0, 0.0, 1.0);
}

// AgX: maps unbounded light to the screen the way film does, keeping hues
// as colours get brighter instead of skewing them. The curve is a sixth
// order fit of AgX's, after Benjamin Wrensch's "Minimal AgX".
fn agx(color: vec3<f32>) -> vec3<f32> {
    let inset = mat3x3<f32>(
        vec3<f32>(0.842479062253094, 0.0423282422610123, 0.0423756549057051),
        vec3<f32>(0.0784335999999992, 0.878468636469772, 0.0784336),
        vec3<f32>(0.0792237451477643, 0.0791661274605434, 0.879142973793104),
    );
    let outset = mat3x3<f32>(
        vec3<f32>(1.19687900512017, -0.0528968517574562, -0.0529716355144438),
        vec3<f32>(-0.0980208811401368, 1.15190312990417, -0.0980434501171241),
        vec3<f32>(-0.0990297440797205, -0.0989611768448433, 1.15107367264116),
    );
    let min_ev = -12.47393;
    let max_ev = 4.026069;
    var x = inset * max(color, vec3<f32>(1e-10));
    x = (clamp(log2(x), vec3<f32>(min_ev), vec3<f32>(max_ev)) - min_ev) / (max_ev - min_ev);
    let x2 = x * x;
    let x4 = x2 * x2;
    x = 15.5 * x4 * x2 - 40.14 * x4 * x + 31.96 * x4 - 6.868 * x2 * x + 0.4298 * x2 + 0.1191 * x
        - 0.00232;
    // A little more contrast and colour than plain AgX, for a game.
    x = pow(max(x, vec3<f32>(0.0)), vec3<f32>(1.35));
    let grey = luminance(x);
    x = grey + (x - grey) * 1.3;
    // Display-encoded, close to sRGB.
    return clamp(outset * x, vec3<f32>(0.0), vec3<f32>(1.0));
}

@fragment
fn composite(in: Fullscreen) -> @location(0) vec4<f32> {
    let frame = textureSampleLevel(source, linear_sampler, in.uv, 0.0).rgb;
    let glow = textureSampleLevel(bloom, linear_sampler, in.uv, 0.0).rgb;
    let exposed = mix(frame, glow, post.frame.z) * textureLoad(exposure, vec2<i32>(0), 0).r;
    var color = agx(exposed);

    if post.overlay.x > 0.5 {
        // Two bars, 2 by 18 pixels, across the middle of the screen.
        let from_middle = abs(in.clip.xy - post.overlay.yz * 0.5);
        if (from_middle.x < 1.0 && from_middle.y < 9.0) || (from_middle.x < 9.0 && from_middle.y < 1.0) {
            color = vec3<f32>(0.97);
        }
    }
    if post.frame.w < 0.5 {
        // The target encodes sRGB on its own: hand it linear light.
        color = pow(color, vec3<f32>(2.2));
    }
    return vec4<f32>(color, 1.0);
}
