//! Window, input and the per-frame game loop, built on `winit` 0.30's
//! `ApplicationHandler`.

use std::sync::Arc;
use std::time::Instant;

use ahash::AHashSet;
use glam::Vec3;
use winit::application::ApplicationHandler;
use winit::dpi::PhysicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, MouseButton, MouseScrollDelta, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Window, WindowId};

use super::{Renderer, CHUNK_EDGE};
use crate::camera::Camera;
use crate::{Game, StreamConfig};
use pixelcraft_physics::MovementInput;

/// Launch the windowed game with the given world seed.
pub fn run(seed: u64) {
    let event_loop = EventLoop::new().expect("create event loop");
    event_loop.set_control_flow(ControlFlow::Poll);
    let mut app = App::new(seed);
    event_loop.run_app(&mut app).expect("event loop");
}

struct App {
    seed: u64,
    state: Option<State>,
}

impl App {
    fn new(seed: u64) -> Self {
        Self { seed, state: None }
    }
}

struct State {
    window: Arc<Window>,
    renderer: Renderer,
    game: Game,
    keys: AHashSet<KeyCode>,
    last_frame: Instant,
    mouse_grabbed: bool,
    // Edge-triggered interaction so holding the button doesn't spam edits.
    mine_queued: bool,
    place_queued: bool,
    third_person: bool,
    fps_accum: f32,
    fps_frames: u32,
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.state.is_some() {
            return;
        }
        let attrs = Window::default_attributes()
            .with_title("PixelCraft — cosy voxel cats 🐈")
            .with_inner_size(PhysicalSize::new(1280, 720));
        let window = Arc::new(event_loop.create_window(attrs).expect("create window"));

        let renderer = pollster::block_on(Renderer::new(window.clone()));

        let config = StreamConfig::default();
        let mut renderer = renderer;
        renderer.set_render_distance(config.view_distance as f32 * CHUNK_EDGE);
        let game = Game::new(self.seed, config, num_workers());

        // Grab the cursor for FPV mouse-look.
        let mouse_grabbed = grab_cursor(&window, true);

        self.state = Some(State {
            window,
            renderer,
            game,
            keys: AHashSet::new(),
            last_frame: Instant::now(),
            mouse_grabbed,
            mine_queued: false,
            place_queued: false,
            third_person: false,
            fps_accum: 0.0,
            fps_frames: 0,
        });
    }

    fn window_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: WindowId,
        event: WindowEvent,
    ) {
        let Some(state) = self.state.as_mut() else {
            return;
        };
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                state.renderer.resize(size.width, size.height);
            }
            WindowEvent::KeyboardInput { event, .. } => {
                if let PhysicalKey::Code(code) = event.physical_key {
                    match event.state {
                        ElementState::Pressed => {
                            if code == KeyCode::Escape {
                                event_loop.exit();
                                return;
                            }
                            // Toggle the crafting menu (frees the cursor).
                            if code == KeyCode::KeyC {
                                state.game.crafting_open = !state.game.crafting_open;
                                state.mouse_grabbed =
                                    grab_cursor(&state.window, !state.game.crafting_open);
                            }
                            // Number keys: craft when the menu is open, else
                            // select a hotbar slot.
                            if let Some(slot) = digit_slot(code) {
                                if state.game.crafting_open {
                                    state.game.craft(slot);
                                } else {
                                    state.game.inventory.select(slot);
                                }
                            }
                            // Talk to a nearby friendly cat.
                            if code == KeyCode::KeyE {
                                state.game.talk();
                            }
                            // Quick save / load.
                            if code == KeyCode::F5 {
                                state.save_game();
                            }
                            if code == KeyCode::F9 {
                                state.load_game();
                            }
                            // Toggle first/third-person view.
                            if code == KeyCode::KeyV {
                                state.third_person = !state.third_person;
                            }
                            state.keys.insert(code);
                        }
                        ElementState::Released => {
                            state.keys.remove(&code);
                        }
                    }
                }
            }
            WindowEvent::MouseInput { state: btn_state, button, .. } => {
                if btn_state == ElementState::Pressed {
                    match button {
                        MouseButton::Left => state.mine_queued = true,
                        MouseButton::Right => state.place_queued = true,
                        _ => {}
                    }
                    // Re-grab if focus was lost.
                    if !state.mouse_grabbed {
                        state.mouse_grabbed = grab_cursor(&state.window, true);
                    }
                }
            }
            WindowEvent::MouseWheel { delta, .. } => {
                let scroll = match delta {
                    MouseScrollDelta::LineDelta(_, y) => y,
                    MouseScrollDelta::PixelDelta(p) => p.y as f32,
                };
                if scroll.abs() > 0.0 {
                    state.game.inventory.scroll(if scroll > 0.0 { -1 } else { 1 });
                }
            }
            WindowEvent::RedrawRequested => {
                state.frame(event_loop);
            }
            WindowEvent::Focused(false) => {
                state.keys.clear();
            }
            _ => {}
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
        _id: DeviceId,
        event: DeviceEvent,
    ) {
        if let DeviceEvent::MouseMotion { delta } = event {
            if let Some(state) = self.state.as_mut() {
                if state.mouse_grabbed {
                    const SENS: f32 = 0.0022;
                    state
                        .game
                        .player
                        .apply_look(delta.0 as f32 * SENS, -delta.1 as f32 * SENS);
                }
            }
        }
    }

    fn about_to_wait(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(state) = self.state.as_ref() {
            state.window.request_redraw();
        }
    }
}

const SAVE_PATH: &str = "pixelcraft_save.dat";

impl State {
    fn save_game(&self) {
        let blob = crate::persistence::save_to_bytes(&self.game);
        match std::fs::write(SAVE_PATH, blob) {
            Ok(()) => println!("Saved to {SAVE_PATH}"),
            Err(e) => eprintln!("Save failed: {e}"),
        }
    }

    fn load_game(&mut self) {
        match std::fs::read(SAVE_PATH) {
            Ok(bytes) => match crate::persistence::load_from_bytes(&bytes, num_workers()) {
                Some(game) => {
                    self.game = game;
                    println!("Loaded {SAVE_PATH}");
                }
                None => eprintln!("Save file is corrupt"),
            },
            Err(e) => eprintln!("Load failed: {e}"),
        }
    }

    /// Distance the third-person camera can pull back from `eye` along `back`
    /// before hitting solid terrain.
    fn third_person_distance(&self, eye: Vec3, back: Vec3, max: f32) -> f32 {
        let world = &self.game.manager.world;
        let registry = &self.game.manager.registry;
        let mut d = 0.4;
        while d < max {
            let p = eye + back * d;
            let bp = pixelcraft_core::coords::BlockPos::new(
                p.x.floor() as i32,
                p.y.floor() as i32,
                p.z.floor() as i32,
            );
            if registry.get(world.block_at(bp)).solid {
                return (d - 0.25).max(0.4);
            }
            d += 0.25;
        }
        max
    }

    fn frame(&mut self, event_loop: &ActiveEventLoop) {
        // Delta time, clamped so a stall doesn't fling the cat across the world.
        let now = Instant::now();
        let dt = (now - self.last_frame).as_secs_f32().min(0.1);
        self.last_frame = now;

        // Build movement intent from the keyboard relative to the look yaw.
        let (forward, right) = self.game.player.horizontal_basis();
        let mut wish = Vec3::ZERO;
        if self.keys.contains(&KeyCode::KeyW) {
            wish += forward;
        }
        if self.keys.contains(&KeyCode::KeyS) {
            wish -= forward;
        }
        if self.keys.contains(&KeyCode::KeyD) {
            wish += right;
        }
        if self.keys.contains(&KeyCode::KeyA) {
            wish -= right;
        }
        let input = MovementInput {
            wish_dir: wish,
            jump: self.keys.contains(&KeyCode::Space),
            sprint: self.keys.contains(&KeyCode::ShiftLeft)
                || self.keys.contains(&KeyCode::ShiftRight),
        };

        self.game.update(input, dt);

        // Resolve queued block interactions against the current look ray
        // (routed through the game so the quest log and inventory update).
        let eye = self.game.player.eye();
        let look = self.game.player.look_dir();
        // Block edits are ignored while the crafting menu is open.
        let mine = std::mem::take(&mut self.mine_queued);
        let place = std::mem::take(&mut self.place_queued);
        if !self.game.crafting_open {
            if mine {
                self.game.do_mine(eye, look);
            }
            if place {
                self.game.do_place(eye, look);
            }
        }

        // Push fresh/edited chunk meshes and the ambient critters to the GPU.
        self.renderer.sync_meshes(&self.game.manager);
        self.renderer.update_entities(&self.game.entities);
        self.renderer.update_particles(&self.game.particles);
        let quest = self.game.quests.hud();
        let hud = super::HudState {
            inventory: &self.game.inventory,
            registry: &self.game.manager.registry,
            time_of_day: self.game.environment.time_of_day,
            objective: quest
                .as_ref()
                .map(|(t, _)| t.clone())
                .or_else(|| Some("ALL QUESTS DONE - ENJOY!".to_string())),
            objective_progress: quest.as_ref().map(|(_, p)| *p),
            dialogue: self.game.active_dialogue.clone(),
            weather: Some({
                let mut w = self.game.weather.label().to_string();
                // Show the firefly tally during the night festival.
                if self.game.environment.daylight() < 0.35 && self.game.fireflies_caught > 0 {
                    w.push_str(&format!("  FIREFLIES {}", self.game.fireflies_caught));
                }
                w
            }),
            crafting: self.game.crafting_open.then(|| {
                crate::crafting::recipes()
                    .iter()
                    .map(|r| (r.name.to_string(), r.affordable(&self.game.inventory)))
                    .collect()
            }),
        };
        self.renderer.update_hud(&hud);

        // Drive vertex animation (water/grass) from elapsed sim time.
        self.renderer.set_time(self.game.time);

        // Camera: first-person at the eye, or pulled back behind the cat in
        // third-person (where we also draw the player's own ginger tabby).
        let camera = if self.third_person {
            // Pull the camera back behind the cat, but stop short of any wall so
            // it never clips inside terrain.
            let back = (-look + Vec3::Y * 0.15).normalize();
            let dist = self.third_person_distance(eye, back, 4.0);
            let cam_eye = eye + back * dist;
            self.renderer
                .update_player_model(Some((self.game.player.position, self.game.player.yaw)));
            Camera::new(cam_eye, look, self.renderer.aspect())
        } else {
            self.renderer.update_player_model(None);
            Camera::new(eye, look, self.renderer.aspect())
        };

        match self.renderer.render(&camera, &self.game.environment) {
            Ok(()) => {}
            Err(wgpu::SurfaceError::Lost | wgpu::SurfaceError::Outdated) => {
                let size = self.window.inner_size();
                self.renderer.resize(size.width, size.height);
            }
            Err(wgpu::SurfaceError::OutOfMemory) => {
                eprintln!("GPU out of memory; exiting.");
                event_loop.exit();
            }
            Err(e) => eprintln!("frame error: {e:?}"),
        }

        // Periodic title-bar HUD.
        self.fps_accum += dt;
        self.fps_frames += 1;
        if self.fps_accum >= 0.5 {
            let fps = self.fps_frames as f32 / self.fps_accum;
            let stats = self.game.manager.stats();
            let held = self.game.inventory.selected_block();
            let held_name = self.game.manager.registry.get(held).name;
            self.window.set_title(&format!(
                "PixelCraft 🐈 | {fps:>5.1} fps | chunks {} (gpu {}) | mem {} KiB | holding: {held_name}",
                stats.loaded_chunks,
                self.renderer.gpu_chunk_count(),
                stats.voxel_memory_bytes / 1024,
            ));
            self.fps_accum = 0.0;
            self.fps_frames = 0;
        }
    }
}

fn grab_cursor(window: &Window, grab: bool) -> bool {
    if grab {
        let ok = window
            .set_cursor_grab(CursorGrabMode::Locked)
            .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined))
            .is_ok();
        window.set_cursor_visible(false);
        ok
    } else {
        let _ = window.set_cursor_grab(CursorGrabMode::None);
        window.set_cursor_visible(true);
        false
    }
}

fn digit_slot(code: KeyCode) -> Option<usize> {
    Some(match code {
        KeyCode::Digit1 => 0,
        KeyCode::Digit2 => 1,
        KeyCode::Digit3 => 2,
        KeyCode::Digit4 => 3,
        KeyCode::Digit5 => 4,
        KeyCode::Digit6 => 5,
        KeyCode::Digit7 => 6,
        KeyCode::Digit8 => 7,
        _ => return None,
    })
}

fn num_workers() -> usize {
    std::thread::available_parallelism()
        .map(|n| (n.get().saturating_sub(1)).max(1))
        .unwrap_or(2)
}
