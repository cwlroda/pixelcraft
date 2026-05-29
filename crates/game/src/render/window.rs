//! Windowed renderer: wraps [`GpuScene`] with a `winit` surface and depth
//! buffer, presenting frames to the screen.

use std::sync::Arc;

use super::scene::{create_depth, request_device, GpuScene};
use crate::camera::Camera;
use crate::environment::Environment;
use crate::streaming::ChunkManager;

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    config: wgpu::SurfaceConfiguration,
    depth_view: wgpu::TextureView,
    scene: GpuScene,
}

impl Renderer {
    pub async fn new(window: Arc<winit::window::Window>) -> Self {
        let size = window.inner_size();
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            ..Default::default()
        });
        let surface = instance
            .create_surface(window.clone())
            .expect("create surface");
        let (adapter, device, queue) = request_device(&instance, Some(&surface)).await;

        let caps = surface.get_capabilities(&adapter);
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width: size.width.max(1),
            height: size.height.max(1),
            present_mode: wgpu::PresentMode::AutoVsync,
            alpha_mode: caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };
        surface.configure(&device, &config);
        let depth_view = create_depth(&device, config.width, config.height);
        let scene = GpuScene::new(device, queue, format);

        Self {
            surface,
            config,
            depth_view,
            scene,
        }
    }

    pub fn set_render_distance(&mut self, d: f32) {
        self.scene.render_distance = d;
    }

    pub fn resize(&mut self, width: u32, height: u32) {
        if width == 0 || height == 0 {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.scene.device, &self.config);
        self.depth_view = create_depth(&self.scene.device, width, height);
    }

    pub fn aspect(&self) -> f32 {
        self.config.width as f32 / self.config.height.max(1) as f32
    }

    pub fn gpu_chunk_count(&self) -> usize {
        self.scene.gpu_chunk_count()
    }

    pub fn sync_meshes(&mut self, manager: &ChunkManager) {
        self.scene.sync_meshes(manager);
    }

    pub fn render(&mut self, camera: &Camera, env: &Environment) -> Result<(), wgpu::SurfaceError> {
        let frame = self.surface.get_current_texture()?;
        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        self.scene.update_globals(camera, env);
        let frustum = camera.frustum();
        let mut encoder =
            self.scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("frame-encoder"),
                });
        self.scene
            .encode(&mut encoder, &view, &self.depth_view, env.sky_color(), &frustum);
        self.scene.queue.submit(Some(encoder.finish()));
        frame.present();
        Ok(())
    }
}
