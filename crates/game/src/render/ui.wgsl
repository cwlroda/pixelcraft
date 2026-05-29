// Flat 2D overlay shader for the HUD (crosshair, hotbar). Positions are already
// in normalised device coordinates; colour is passed straight through with
// alpha blending. No lighting, no depth interaction.

struct VsIn {
    @location(0) pos : vec2<f32>,
    @location(1) color : vec4<f32>,
};

struct VsOut {
    @builtin(position) clip_pos : vec4<f32>,
    @location(0) color : vec4<f32>,
};

@vertex
fn vs_main(in : VsIn) -> VsOut {
    var out : VsOut;
    out.clip_pos = vec4<f32>(in.pos, 0.0, 1.0);
    out.color = in.color;
    return out;
}

@fragment
fn fs_main(in : VsOut) -> @location(0) vec4<f32> {
    return in.color;
}
