//! GPU rendering front-end built on `wgpu`.
//!
//! * [`scene::GpuScene`] is the backend-agnostic core (device, pipelines,
//!   per-chunk mesh buffers, draw encoding).
//! * The windowed renderer (`render` feature) presents to a `winit` surface.
//! * The headless capturer (`capture` feature) renders offscreen to a PNG,
//!   used to screenshot the game in environments without a display.

mod font;
mod scene;
mod textures;
mod ui;
pub use scene::CHUNK_EDGE;
pub use ui::HudState;

#[cfg(feature = "render")]
mod app;
#[cfg(feature = "render")]
mod window;
#[cfg(feature = "render")]
pub use app::run;
#[cfg(feature = "render")]
pub use window::Renderer;

#[cfg(feature = "capture")]
mod capture;
#[cfg(feature = "capture")]
pub use capture::Headless;
