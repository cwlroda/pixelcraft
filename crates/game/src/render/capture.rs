//! Headless offscreen renderer: draws the world into an off-screen texture and
//! writes it to a PNG. Used to screenshot the game where there is no display
//! (rendered via a software Vulkan device such as Mesa's lavapipe).

use std::path::Path;

use super::scene::{create_depth, request_device, GpuScene};
use crate::camera::Camera;
use crate::environment::Environment;
use crate::streaming::ChunkManager;

const FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba8UnormSrgb;

fn align_up(value: u32, align: u32) -> u32 {
    ((value + align - 1) / align) * align
}

pub struct Headless {
    scene: GpuScene,
    width: u32,
    height: u32,
    color_texture: wgpu::Texture,
    color_view: wgpu::TextureView,
    depth_view: wgpu::TextureView,
    output_buffer: wgpu::Buffer,
    padded_bytes_per_row: u32,
}

impl Headless {
    /// Create an offscreen renderer at the given resolution.
    pub fn new(width: u32, height: u32) -> Self {
        pollster::block_on(Self::new_async(width, height))
    }

    async fn new_async(width: u32, height: u32) -> Self {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            // Allow all backends so software Vulkan (lavapipe) is discoverable.
            backends: wgpu::Backends::all(),
            ..Default::default()
        });
        let (_adapter, device, queue) = request_device(&instance, None).await;

        let color_texture = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("offscreen-color"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: FORMAT,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let color_view = color_texture.create_view(&wgpu::TextureViewDescriptor::default());
        let depth_view = create_depth(&device, width, height);

        let padded_bytes_per_row = align_up(width * 4, wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let output_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: (padded_bytes_per_row * height) as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let scene = GpuScene::new(device, queue, FORMAT);

        Self {
            scene,
            width,
            height,
            color_texture,
            color_view,
            depth_view,
            output_buffer,
            padded_bytes_per_row,
        }
    }

    pub fn set_render_distance(&mut self, d: f32) {
        self.scene.render_distance = d;
    }

    pub fn aspect(&self) -> f32 {
        self.width as f32 / self.height.max(1) as f32
    }

    pub fn gpu_chunk_count(&self) -> usize {
        self.scene.gpu_chunk_count()
    }

    pub fn sync_meshes(&mut self, manager: &ChunkManager) {
        self.scene.sync_meshes(manager);
    }

    /// Render one frame and save it as a PNG at `path`.
    pub fn capture(&self, camera: &Camera, env: &Environment, path: impl AsRef<Path>) {
        self.scene.update_globals(camera, env);
        let frustum = camera.frustum();

        let mut encoder =
            self.scene
                .device
                .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("capture-encoder"),
                });
        self.scene.encode(
            &mut encoder,
            &self.color_view,
            &self.depth_view,
            env.sky_color(),
            &frustum,
        );
        // Copy the rendered texture into the mappable readback buffer.
        encoder.copy_texture_to_buffer(
            wgpu::ImageCopyTexture {
                texture: &self.color_texture,
                mip_level: 0,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            wgpu::ImageCopyBuffer {
                buffer: &self.output_buffer,
                layout: wgpu::ImageDataLayout {
                    offset: 0,
                    bytes_per_row: Some(self.padded_bytes_per_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        self.scene.queue.submit(Some(encoder.finish()));

        // Map and read back, stripping row padding.
        let slice = self.output_buffer.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.scene.device.poll(wgpu::Maintain::Wait);
        rx.recv().unwrap().expect("map readback buffer");

        let data = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((self.width * self.height * 4) as usize);
        for row in 0..self.height {
            let start = (row * self.padded_bytes_per_row) as usize;
            let end = start + (self.width * 4) as usize;
            pixels.extend_from_slice(&data[start..end]);
        }
        drop(data);
        self.output_buffer.unmap();

        write_png(path.as_ref(), self.width, self.height, &pixels);
    }
}

fn write_png(path: &Path, width: u32, height: u32, rgba: &[u8]) {
    if let Some(parent) = path.parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let file = std::fs::File::create(path).expect("create png file");
    let w = std::io::BufWriter::new(file);
    let mut encoder = png::Encoder::new(w, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    // The texture is sRGB, so the bytes are already gamma-encoded.
    encoder.set_source_srgb(png::SrgbRenderingIntent::Perceptual);
    let mut writer = encoder.write_header().expect("png header");
    writer.write_image_data(rgba).expect("write png data");
}
