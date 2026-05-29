// World shader: textured-by-vertex-colour voxel terrain with soft directional
// lighting, a warm cosy ambient term, and distance fog that melts into the sky
// colour so the streaming frontier is never a hard pop-in edge.

struct Globals {
    view_proj : mat4x4<f32>,
    camera_pos : vec4<f32>,   // xyz = eye, w = render distance (world units)
    sun_dir : vec4<f32>,      // xyz = direction TO the sun (normalised)
    sky_color : vec4<f32>,    // rgb = horizon/fog colour
    sun_color : vec4<f32>,    // rgb = sun tint, a = ambient strength
};

@group(0) @binding(0) var<uniform> globals : Globals;
@group(0) @binding(1) var atlas : texture_2d_array<f32>;
@group(0) @binding(2) var atlas_sampler : sampler;

// Chunk meshes are uploaded with world-space positions baked in (chunks never
// move), so the vertex stream is self-contained. `color` is a luminance
// multiplier (face shade × AO); the tint comes from the texture `layer` sampled
// at the tiling `uv`.
struct VsIn {
    @location(0) position : vec3<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) color : vec4<f32>,
    @location(3) uv : vec2<f32>,
    @location(4) layer : u32,
};

struct VsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) color : vec4<f32>,
    @location(1) normal : vec3<f32>,
    @location(2) world_pos : vec3<f32>,
    @location(3) uv : vec2<f32>,
    @location(4) @interpolate(flat) layer : u32,
};

@vertex
fn vs_main(in : VsIn) -> VsOut {
    var out : VsOut;
    out.world_pos = in.position;
    out.clip_pos = globals.view_proj * vec4<f32>(in.position, 1.0);
    out.color = in.color;
    out.normal = in.normal;
    out.uv = in.uv;
    out.layer = in.layer;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    // Tile the per-block texture across greedy-merged quads.
    let texel = textureSample(atlas, atlas_sampler, fract(in.uv), i32(in.layer));
    // Albedo = texture tint × baked luminance (face shade × AO).
    let albedo = texel.rgb * in.color.rgb;
    let alpha = texel.a * in.color.a;

    let n = normalize(in.normal);
    let sun = normalize(globals.sun_dir.xyz);

    // Lambertian sun term, lifted by a soft ambient so shadowed faces stay
    // readable and cosy rather than crushed to black.
    let ambient = globals.sun_color.a;
    let diffuse = max(dot(n, sun), 0.0);
    let light = ambient + (1.0 - ambient) * diffuse;
    var lit = albedo * light * globals.sun_color.rgb;

    // Gentle hemispherical sky bounce on upward faces for extra warmth.
    let sky_bounce = clamp(n.y * 0.5 + 0.5, 0.0, 1.0) * 0.08;
    lit += globals.sky_color.rgb * sky_bounce;

    // Distance fog → sky colour.
    let dist = length(in.world_pos - globals.camera_pos.xyz);
    let fog_start = globals.camera_pos.w * 0.55;
    let fog_end = globals.camera_pos.w * 0.98;
    let fog = clamp((dist - fog_start) / max(fog_end - fog_start, 1.0), 0.0, 1.0);
    let final_rgb = mix(lit, globals.sky_color.rgb, fog);

    return vec4<f32>(final_rgb, alpha);
}
