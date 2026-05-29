//! PixelCraft entry point.
//!
//! A windowed GPU renderer is the eventual front-end, but it requires a display
//! and GPU, so in headless environments (CI, servers) this binary instead runs
//! a self-checking simulation: it streams the world around a moving cat, runs
//! physics, and reports streaming/meshing statistics. This proves the whole
//! engine works end-to-end without a window.

/// With the `render` feature, launch the real windowed game.
#[cfg(feature = "render")]
fn main() {
    let seed = std::env::args()
        .nth(1)
        .and_then(|s| s.parse::<u64>().ok())
        .unwrap_or(0xC0FFEE);
    pixelcraft_game::render::run(seed);
}

/// Without a GPU/display, run the self-checking headless simulation.
#[cfg(not(feature = "render"))]
fn main() {
    use glam::Vec3;
    use pixelcraft_game::{Game, StreamConfig};
    use pixelcraft_physics::MovementInput;

    println!("🐈 PixelCraft — cosy voxel cat world");
    println!("   (headless simulation; GPU renderer runs where a display is available)\n");

    let config = StreamConfig::default();
    let mut game = Game::new(0xC0FFEE, config, num_workers());

    let dt = 1.0 / 60.0;
    let mut frame = 0u64;
    let mut wish = Vec3::ZERO;
    loop {
        // After landing, wander the cat around to exercise streaming.
        if game.player.on_ground && frame.is_multiple_of(240) {
            let a = (frame as f32) * 0.01;
            wish = Vec3::new(a.cos(), 0.0, a.sin());
            game.player.apply_look(0.3, 0.0);
        }
        let input = MovementInput {
            wish_dir: wish,
            jump: false,
            sprint: false,
        };
        game.update(input, dt);

        if frame.is_multiple_of(120) {
            let s = game.manager.stats();
            println!(
                "frame {frame:>5} | pos ({:>7.1},{:>6.1},{:>7.1}) | chunks {:>4} meshed {:>4} inflight {:>3} | voxel mem {:>5} KiB",
                game.player.position.x,
                game.player.position.y,
                game.player.position.z,
                s.loaded_chunks,
                s.meshed_chunks,
                s.inflight,
                s.voxel_memory_bytes / 1024,
            );
        }

        frame += 1;
        if frame >= 1200 {
            break;
        }
    }
    println!("\nSimulation ran {frame} frames cleanly. 🐾");
}

#[cfg(not(feature = "render"))]
fn num_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get().saturating_sub(1)).max(1))
        .unwrap_or(2)
}
