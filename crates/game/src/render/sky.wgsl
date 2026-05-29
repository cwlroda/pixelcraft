// Procedural sky: a horizon→zenith gradient with a warm sun disk, a soft moon
// at night, and a sprinkle of stars that fade in after dusk. Drawn as a single
// full-screen triangle behind the world; the per-fragment view ray is
// reconstructed from the inverse view-projection.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,
    sun_dir : vec4<f32>,
    sky_color : vec4<f32>,    // horizon colour
    sun_color : vec4<f32>,    // a = ambient strength
    sky_zenith : vec4<f32>,   // overhead colour
    inv_view_proj : mat4x4<f32>,
};

@group(0) @binding(0) var<uniform> globals : Globals;

struct VsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) ndc : vec2<f32>,
};

@vertex
fn vs_main(@builtin(vertex_index) vi : u32) -> VsOut {
    // Full-screen triangle.
    var pts = array<vec2<f32>, 3>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>(3.0, -1.0),
        vec2<f32>(-1.0, 3.0),
    );
    var out : VsOut;
    let p = pts[vi];
    out.clip_pos = vec4<f32>(p, 1.0, 1.0);
    out.ndc = p;
    return out;
}

// Cheap hash for star placement.
fn hash21(p : vec2<f32>) -> f32 {
    var h = fract(sin(dot(p, vec2<f32>(127.1, 311.7))) * 43758.5453);
    return h;
}

// Value noise + fBm for soft clouds.
fn vnoise(p : vec2<f32>) -> f32 {
    let i = floor(p);
    let f = fract(p);
    let a = hash21(i);
    let b = hash21(i + vec2<f32>(1.0, 0.0));
    let c = hash21(i + vec2<f32>(0.0, 1.0));
    let d = hash21(i + vec2<f32>(1.0, 1.0));
    let u = f * f * (3.0 - 2.0 * f);
    return mix(mix(a, b, u.x), mix(c, d, u.x), u.y);
}

fn fbm(p : vec2<f32>) -> f32 {
    var sum = 0.0;
    var amp = 0.5;
    var freq = p;
    for (var i = 0; i < 4; i = i + 1) {
        sum = sum + vnoise(freq) * amp;
        freq = freq * 2.02;
        amp = amp * 0.5;
    }
    return sum;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    // Reconstruct the world-space view ray for this pixel.
    let far = globals.inv_view_proj * vec4<f32>(in.ndc, 1.0, 1.0);
    let world = far.xyz / far.w;
    let dir = normalize(world - globals.camera_pos.xyz);

    // Horizon → zenith gradient by altitude.
    let t = clamp(dir.y * 0.5 + 0.5, 0.0, 1.0);
    var col = mix(globals.sky_color.rgb, globals.sky_zenith.rgb, pow(t, 0.8));

    let sun = normalize(globals.sun_dir.xyz);
    let daylight = clamp(sun.y * 1.5 + 0.3, 0.0, 1.0);

    // Stars: sparse twinkles in the sky, only visible at night and above the
    // horizon. Quantise the direction into a grid and threshold the hash.
    if dir.y > 0.02 {
        let grid = floor(dir.xz / max(dir.y, 0.05) * 60.0);
        let star = hash21(grid);
        if star > 0.985 {
            let twinkle = hash21(grid + 3.0);
            col += vec3<f32>(1.0) * (1.0 - daylight) * (0.6 + 0.4 * twinkle);
        }
    }

    // Drifting clouds: project the ray onto a high plane and sample fBm,
    // scrolling slowly with time. Fades out toward the horizon.
    if dir.y > 0.03 {
        let t = globals.sun_dir.w * 0.006;
        let proj = dir.xz / dir.y;
        let uv = proj * 0.6 + vec2<f32>(t, t * 0.5);
        let n = fbm(uv);
        let cover = smoothstep(0.52, 0.78, n) * smoothstep(0.03, 0.35, dir.y);
        // Clouds tinted by the sky: bright by day, dusky at night.
        let cloud_col = mix(globals.sky_color.rgb, vec3<f32>(1.0, 0.98, 0.96), 0.7)
            * (0.45 + 0.55 * daylight);
        col = mix(col, cloud_col, cover * 0.9);
    }

    // Sun disk + glow.
    let sd = max(dot(dir, sun), 0.0);
    let disk = smoothstep(0.9975, 0.9990, sd);
    let glow = pow(sd, 64.0) * 0.4;
    col += globals.sun_color.rgb * (disk + glow) * clamp(daylight + 0.2, 0.0, 1.0);

    // Moon: a soft disk opposite the sun, visible at night.
    let md = max(dot(dir, -sun), 0.0);
    let moon = smoothstep(0.9980, 0.9992, md);
    col += vec3<f32>(0.8, 0.85, 1.0) * moon * (1.0 - daylight);

    return vec4<f32>(col, 1.0);
}
