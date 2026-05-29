//! Headless screenshot tool: builds a world, lets the cat settle onto terrain,
//! then renders a set of PNGs from scenic camera angles at different times of
//! day. Runs without a display via a software Vulkan device (lavapipe).
//!
//! Usage: `cargo run --bin screenshot --features capture -- [seed] [out_dir]`

use std::time::Duration;

use glam::Vec3;
use pixelcraft_game::environment::Environment;
use pixelcraft_game::render::{Headless, CHUNK_EDGE};
use pixelcraft_game::{Camera, Game, StreamConfig};
use pixelcraft_physics::MovementInput;

const WIDTH: u32 = 1280;
const HEIGHT: u32 = 720;

fn main() {
    let mut args = std::env::args().skip(1);
    let seed: u64 = args
        .next()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0xCA75_C0DE);
    let out_dir = args.next().unwrap_or_else(|| "screenshots".to_string());

    let config = StreamConfig {
        view_distance: 9,
        min_chunk_y: 0,
        max_chunk_y: 5,
        unload_margin: 2,
        max_uploads_per_update: 64,
        max_meshes_per_update: 48,
        max_jobs_per_update: 256,
    };
    let render_distance = config.view_distance as f32 * CHUNK_EDGE;

    println!("Building world (seed {seed:#x})…");
    let mut game = Game::new(seed, config, 3);

    // Let the cat fall to the ground and the surrounding region stream + mesh in.
    let dt = 1.0 / 60.0;
    for step in 0..4000 {
        game.update(MovementInput::default(), dt);
        if step % 20 == 0 {
            std::thread::sleep(Duration::from_millis(3));
        }
        if game.player.on_ground && game.manager.is_settled() {
            println!("World settled after {step} steps.");
            break;
        }
    }

    let stats = game.manager.stats();
    println!(
        "Loaded {} chunks, {} meshed, voxel memory {} KiB. Cat at {:?}.",
        stats.loaded_chunks,
        stats.meshed_chunks,
        stats.voxel_memory_bytes / 1024,
        game.player.position,
    );

    println!("Initialising offscreen renderer (lavapipe)…");
    let mut renderer = Headless::new(WIDTH, HEIGHT);
    renderer.set_render_distance(render_distance);
    renderer.sync_meshes(&game.manager);
    println!("GPU chunk meshes: {}", renderer.gpu_chunk_count());

    let ground = game.player.position;
    let aspect = renderer.aspect();

    // A handful of scenic shots. Each entry: (filename, eye, yaw, pitch, env).
    let eye_level = ground + Vec3::new(0.0, 1.4, 0.0);
    let drone = ground + Vec3::new(0.0, 22.0, 0.0);

    let shots: &[(&str, Vec3, f32, f32, f32)] = &[
        // name, eye, yaw, pitch(rad), time_of_day
        ("01_morning_eye.png", eye_level, 0.6, -0.12, 0.10),
        ("02_morning_vista.png", drone, 0.9, -0.62, 0.12),
        ("03_noon_vista.png", drone, 2.3, -0.55, 0.25),
        ("04_golden_hour.png", drone, 4.0, -0.42, 0.46),
        ("05_night.png", drone, 5.2, -0.5, 0.72),
    ];

    for (name, eye, yaw, pitch, tod) in shots {
        let env = Environment {
            time_of_day: *tod,
            day_length: 600.0,
        };
        let forward = look_dir(*yaw, *pitch);
        let camera = Camera::new(*eye, forward, aspect);
        let path = format!("{out_dir}/{name}");
        renderer.capture(&camera, &env, &path);
        println!("  wrote {path}");
    }

    println!("Done. Screenshots in '{out_dir}/'.");
}

/// Forward direction from yaw/pitch (matches Player::look_dir).
fn look_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(cp * sy, sp, -cp * cy).normalize()
}
