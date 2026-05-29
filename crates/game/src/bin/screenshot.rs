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

    // Let ambient critters populate and wander a little before the photo shoot.
    for _ in 0..220 {
        game.update(MovementInput::default(), dt);
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
    // Give the cat a few starter blocks so the hotbar shows nicely.
    for &id in &pixelcraft_game::HOTBAR {
        game.inventory.add(id, 24);
    }

    renderer.set_render_distance(render_distance);
    renderer.set_time(game.time);
    renderer.sync_meshes(&game.manager);
    renderer.update_entities(&game.entities);
    let quest = game.quests.hud();
    let hud = pixelcraft_game::render::HudState {
        inventory: &game.inventory,
        registry: &game.manager.registry,
        time_of_day: 0.12,
        objective: quest.as_ref().map(|(t, _)| t.clone()),
        objective_progress: quest.as_ref().map(|(_, p)| *p),
        dialogue: None,
    };
    renderer.update_hud(&hud);
    println!(
        "GPU chunk meshes: {} | critters: {}",
        renderer.gpu_chunk_count(),
        game.entities.count()
    );

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

    // If a cottage generated nearby, frame it for a close-up.
    if let Some(cottage) = nearest_block(&game, ground, pixelcraft_core::block::blocks::ROOF) {
        // Clear the trees immediately around the house so the architecture is
        // visible for the close-up (cosmetic, screenshot-only).
        use pixelcraft_core::block::{blocks, BlockId};
        use pixelcraft_core::coords::BlockPos;
        for dx in -9..=9 {
            for dz in -9..=9 {
                for dy in -2..=12 {
                    let p = BlockPos::new(cottage.x + dx, cottage.y + dy, cottage.z + dz);
                    // Keep blocks belonging to the house footprint.
                    if dx.abs() <= 3 && dz.abs() <= 3 && dy >= -6 {
                        continue;
                    }
                    let b = game.manager.world.block_at(p);
                    if b == blocks::TRUNK || b == blocks::LEAVES {
                        game.manager.set_block(p, BlockId::AIR);
                    }
                }
            }
        }
        // Remesh the edits, then refresh GPU buffers.
        for _ in 0..6 {
            game.manager.update(ground);
        }
        renderer.sync_meshes(&game.manager);

        let center = Vec3::new(cottage.x as f32, cottage.y as f32 - 3.0, cottage.z as f32);
        let eye = center + Vec3::new(8.0, 6.0, 8.0);
        let forward = (center - eye).normalize();
        let env = Environment {
            time_of_day: 0.16,
            day_length: 600.0,
        };
        let camera = Camera::new(eye, forward, aspect);
        let path = format!("{out_dir}/06_cottage.png");
        renderer.capture(&camera, &env, &path);
        println!("  wrote {path} (cottage at {cottage:?})");
    } else {
        println!("  (no cottage found near spawn this run)");
    }

    // If a cherry grove is near, frame its pink canopy with drifting petals.
    if let Some(cherry) = nearest_block(&game, ground, pixelcraft_core::block::blocks::CHERRY_LEAVES) {
        // Look down over the grove from above the canopy.
        let center = Vec3::new(cherry.x as f32, cherry.y as f32 - 4.0, cherry.z as f32);
        let eye = center + Vec3::new(16.0, 16.0, 16.0);
        let forward = (center - eye).normalize();
        // Seed a flurry of petals around the grove for the shot.
        let grove = Vec3::new(cherry.x as f32, cherry.y as f32, cherry.z as f32);
        for _ in 0..120 {
            game.particles.emit_petals(grove, 0.1);
            game.particles.update(0.12);
        }
        renderer.update_particles(&game.particles);
        let env = Environment { time_of_day: 0.14, day_length: 600.0 };
        let camera = Camera::new(eye, forward, aspect);
        let path = format!("{out_dir}/08_cherry.png");
        renderer.capture(&camera, &env, &path);
        println!("  wrote {path} (cherry at {cherry:?})");
    } else {
        println!("  (no cherry grove near spawn this run)");
    }

    // A critter close-up: frame the nearest ground critter to the cat.
    if let Some(critter) = game
        .entities
        .entities
        .iter()
        .filter(|e| e.kind == pixelcraft_game::EntityKind::Critter)
        .min_by(|a, b| {
            a.position
                .distance_squared(ground)
                .total_cmp(&b.position.distance_squared(ground))
        })
        .map(|e| e.position)
    {
        let eye = critter + Vec3::new(2.2, 1.1, 2.2);
        let forward = (critter + Vec3::new(0.0, 0.3, 0.0) - eye).normalize();
        let env = Environment {
            time_of_day: 0.16,
            day_length: 600.0,
        };
        let camera = Camera::new(eye, forward, aspect);
        let path = format!("{out_dir}/07_critter.png");
        renderer.capture(&camera, &env, &path);
        println!("  wrote {path}");
    }

    // A friendly cat NPC with a dialogue box.
    if let Some((pos, name, line)) = game
        .entities
        .entities
        .iter()
        .filter(|e| e.kind == pixelcraft_game::EntityKind::Friend)
        .min_by(|a, b| {
            a.position
                .distance_squared(ground)
                .total_cmp(&b.position.distance_squared(ground))
        })
        .and_then(|e| {
            e.npc
                .as_ref()
                .map(|n| (e.position, n.name.to_string(), n.lines[0].to_string()))
        })
    {
        let center = pos + Vec3::new(0.0, 0.5, 0.0);
        let eye = center + Vec3::new(1.8, 0.6, 1.8);
        let forward = (center - eye).normalize();
        let env = Environment { time_of_day: 0.15, day_length: 600.0 };
        let camera = Camera::new(eye, forward, aspect);
        let hud = pixelcraft_game::render::HudState {
            inventory: &game.inventory,
            registry: &game.manager.registry,
            time_of_day: 0.15,
            objective: None,
            objective_progress: None,
            dialogue: Some((name, line)),
        };
        renderer.update_hud(&hud);
        let path = format!("{out_dir}/11_friend.png");
        renderer.capture(&camera, &env, &path);
        println!("  wrote {path}");
    } else {
        println!("  (no friendly cat near spawn this run)");
    }

    // A cosy night scene with glowing lanterns: stack a little lantern post
    // near the cat and shoot it under the stars.
    {
        use pixelcraft_core::block::{blocks, BlockId};
        use pixelcraft_core::coords::BlockPos;
        let gx = ground.x.floor() as i32 + 3;
        let gz = ground.z.floor() as i32;
        let gy = game.manager.surface_height(gx, gz);
        // Two lantern posts on plank pillars.
        for &dx in &[0, 3] {
            for h in 1..=2 {
                game.manager
                    .set_block(BlockPos::new(gx + dx, gy + h, gz), blocks::PLANK);
            }
            game.manager
                .set_block(BlockPos::new(gx + dx, gy + 3, gz), blocks::LANTERN);
        }
        let _ = BlockId::AIR;
        for _ in 0..6 {
            game.manager.update(ground);
        }
        let mut nrender = Headless::new(WIDTH, HEIGHT);
        nrender.set_render_distance(render_distance);
        nrender.set_time(game.time);
        nrender.sync_meshes(&game.manager);
        nrender.update_entities(&game.entities);
        let center = Vec3::new(gx as f32 + 1.5, gy as f32 + 2.5, gz as f32);
        let eye = center + Vec3::new(4.0, 1.5, 5.0);
        let forward = (center - eye).normalize();
        let env = Environment { time_of_day: 0.72, day_length: 600.0 };
        let camera = Camera::new(eye, forward, nrender.aspect());
        let path = format!("{out_dir}/10_lantern_night.png");
        nrender.capture(&camera, &env, &path);
        println!("  wrote {path}");
    }

    // A day→night→day timelapse GIF over the valley.
    make_timelapse(&game, ground, &out_dir);

    println!("Done. Screenshots in '{out_dir}/'.");
}

/// Render a small animated GIF cycling through a full day, showing the moving
/// sky, sun/stars and rippling water.
fn make_timelapse(game: &Game, ground: Vec3, out_dir: &str) {
    const W: u32 = 480;
    const H: u32 = 270;
    const FRAMES: u32 = 40;

    let mut clip = pixelcraft_game::render::Headless::new(W, H);
    clip.set_render_distance(9.0 * CHUNK_EDGE);
    clip.sync_meshes(&game.manager);
    clip.update_entities(&game.entities);

    let center = ground + Vec3::new(0.0, 2.0, 0.0);
    let eye = center + Vec3::new(16.0, 12.0, 16.0);
    let forward = (center - eye).normalize();
    let camera = Camera::new(eye, forward, W as f32 / H as f32);

    let path = format!("{out_dir}/09_daycycle.gif");
    let file = std::fs::File::create(&path).expect("create gif");
    let mut encoder = gif::Encoder::new(file, W as u16, H as u16, &[]).expect("gif encoder");
    encoder.set_repeat(gif::Repeat::Infinite).ok();

    for i in 0..FRAMES {
        let tod = i as f32 / FRAMES as f32;
        let env = Environment { time_of_day: tod, day_length: 600.0 };
        clip.set_time(i as f32 * 0.35);
        let mut rgba = clip.capture_rgba(&camera, &env);
        let mut frame = gif::Frame::from_rgba_speed(W as u16, H as u16, &mut rgba, 10);
        frame.delay = 8; // ~80 ms per frame
        encoder.write_frame(&frame).expect("write gif frame");
    }
    println!("  wrote {path} ({FRAMES} frames)");
}

/// Scan loaded chunks for the nearest voxel of `target` to `near`.
fn nearest_block(
    game: &Game,
    near: Vec3,
    target: pixelcraft_core::block::BlockId,
) -> Option<pixelcraft_core::coords::BlockPos> {
    use pixelcraft_core::coords::{LocalPos, CHUNK_VOLUME};
    let mut best: Option<(f32, pixelcraft_core::coords::BlockPos)> = None;
    for chunk in game.manager.world.iter_chunks() {
        for i in 0..CHUNK_VOLUME {
            if chunk.storage.get_index(i) != target {
                continue;
            }
            let wp = LocalPos::from_index(i).to_block(chunk.pos);
            let d = Vec3::new(wp.x as f32, wp.y as f32, wp.z as f32).distance_squared(near);
            if best.map(|(bd, _)| d < bd).unwrap_or(true) {
                best = Some((d, wp));
            }
        }
    }
    best.map(|(_, p)| p)
}

/// Forward direction from yaw/pitch (matches Player::look_dir).
fn look_dir(yaw: f32, pitch: f32) -> Vec3 {
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    Vec3::new(cp * sy, sp, -cp * cy).normalize()
}
