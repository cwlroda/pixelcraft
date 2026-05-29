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
use crate::interaction;
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
                            // Hotbar number keys 1..8.
                            if let Some(slot) = digit_slot(code) {
                                state.game.inventory.select(slot);
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

impl State {
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

        // Resolve queued block interactions against the current look ray.
        let eye = self.game.player.eye();
        let look = self.game.player.look_dir();
        if std::mem::take(&mut self.mine_queued) {
            interaction::mine(&mut self.game.manager, &mut self.game.inventory, eye, look);
        }
        if std::mem::take(&mut self.place_queued) {
            let body = self.game.player.aabb();
            interaction::place(
                &mut self.game.manager,
                &mut self.game.inventory,
                eye,
                look,
                body,
            );
        }

        // Push fresh/edited chunk meshes and the ambient critters to the GPU.
        self.renderer.sync_meshes(&self.game.manager);
        self.renderer.update_entities(&self.game.entities);

        // Camera follows the eye.
        let camera = Camera::new(eye, look, self.renderer.aspect());

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
