//! Backend-agnostic GPU rendering core shared by the windowed renderer and the
//! headless screenshot capturer. Owns the device, pipelines, the per-chunk mesh
//! buffers and the globals uniform; knows how to upload meshes and encode the
//! opaque+transparent draw, but is told *where* to draw (a surface texture or an
//! offscreen texture) by the caller.

use ahash::AHashMap;
use bytemuck::{Pod, Zeroable};
use glam::Vec3;
use pixelcraft_core::coords::{ChunkPos, CHUNK_SIZE};
use pixelcraft_mesh::Vertex;
use wgpu::util::DeviceExt;

use crate::camera::{Camera, Frustum};
use crate::environment::Environment;
use crate::streaming::ChunkManager;

/// One chunk edge in world units.
pub const CHUNK_EDGE: f32 = CHUNK_SIZE as f32;

pub const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;

/// Uniform block shared with `world.wgsl`. Layout must match the WGSL `Globals`.
#[repr(C)]
#[derive(Clone, Copy, Pod, Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    camera_pos: [f32; 4],
    sun_dir: [f32; 4],
    sky_color: [f32; 4],
    sun_color: [f32; 4],
}

/// A chunk's geometry resident on the GPU, one buffer set per draw layer.
struct GpuLayer {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    aabb_min: Vec3,
    aabb_max: Vec3,
}

struct GpuChunk {
    opaque: Option<GpuLayer>,
    transparent: Option<GpuLayer>,
    /// Mesh revision this GPU data was built from; re-uploaded when it changes.
    version: u64,
}

/// The shared GPU state.
pub struct GpuScene {
    pub device: wgpu::Device,
    pub queue: wgpu::Queue,
    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    opaque_pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    chunks: AHashMap<ChunkPos, GpuChunk>,
    /// Dynamic geometry for ambient critters, rebuilt each frame.
    entities: Option<GpuLayer>,
    ui_pipeline: wgpu::RenderPipeline,
    ui_buffer: Option<(wgpu::Buffer, u32)>,
    pub render_distance: f32,
}

impl GpuScene {
    /// Build pipelines and uniforms for a given colour target format.
    pub fn new(device: wgpu::Device, queue: wgpu::Queue, format: wgpu::TextureFormat) -> Self {
        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let bind_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals-layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals-bind"),
            layout: &bind_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("world-shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("world.wgsl").into()),
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world-layout"),
            bind_group_layouts: &[&bind_layout],
            push_constant_ranges: &[],
        });
        let opaque_pipeline = make_pipeline(&device, &layout, &shader, format, false);
        let transparent_pipeline = make_pipeline(&device, &layout, &shader, format, true);
        let ui_pipeline = make_ui_pipeline(&device, format);

        Self {
            device,
            queue,
            globals_buffer,
            globals_bind_group,
            opaque_pipeline,
            transparent_pipeline,
            ui_pipeline,
            ui_buffer: None,
            chunks: AHashMap::new(),
            entities: None,
            render_distance: 16.0 * CHUNK_EDGE,
        }
    }

    /// Replace the HUD overlay geometry for this frame.
    pub fn upload_ui(&mut self, verts: &[super::ui::UiVertex]) {
        if verts.is_empty() {
            self.ui_buffer = None;
            return;
        }
        let buffer = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("ui-vertices"),
                contents: bytemuck::cast_slice(verts),
                usage: wgpu::BufferUsages::VERTEX,
            });
        self.ui_buffer = Some((buffer, verts.len() as u32));
    }

    /// Replace the per-frame entity geometry (world-space vertices).
    pub fn upload_entities(&mut self, verts: &[Vertex], indices: &[u32]) {
        self.entities = self.upload_layer(verts, indices, Vec3::ZERO);
    }

    pub fn gpu_chunk_count(&self) -> usize {
        self.chunks.len()
    }

    /// Reconcile GPU chunk buffers with the manager's CPU meshes.
    pub fn sync_meshes(&mut self, manager: &ChunkManager) {
        let live: ahash::AHashSet<ChunkPos> = manager.meshes().map(|(p, _)| *p).collect();
        self.chunks.retain(|p, _| live.contains(p));

        // Collect uploads first to avoid borrowing self immutably and mutably.
        // Upload chunks we don't hold yet, or whose mesh revision has changed
        // (e.g. after a block edit), so builds/mining update on screen.
        let mut uploads: Vec<(ChunkPos, Option<GpuLayer>, Option<GpuLayer>, u64)> = Vec::new();
        for (pos, mesh) in manager.meshes() {
            let version = manager.mesh_version(*pos);
            if self.chunks.get(pos).map(|c| c.version) == Some(version) {
                continue;
            }
            let origin = pos.origin();
            let offset = Vec3::new(origin.x as f32, origin.y as f32, origin.z as f32);
            let opaque = self.upload_layer(&mesh.opaque.vertices, &mesh.opaque.indices, offset);
            let transparent =
                self.upload_layer(&mesh.transparent.vertices, &mesh.transparent.indices, offset);
            uploads.push((*pos, opaque, transparent, version));
        }
        for (pos, opaque, transparent, version) in uploads {
            self.chunks.insert(
                pos,
                GpuChunk {
                    opaque,
                    transparent,
                    version,
                },
            );
        }
    }

    fn upload_layer(&self, verts: &[Vertex], indices: &[u32], offset: Vec3) -> Option<GpuLayer> {
        if indices.is_empty() {
            return None;
        }
        let mut min = Vec3::splat(f32::INFINITY);
        let mut max = Vec3::splat(f32::NEG_INFINITY);
        let world_verts: Vec<Vertex> = verts
            .iter()
            .map(|v| {
                let p = [
                    v.position[0] + offset.x,
                    v.position[1] + offset.y,
                    v.position[2] + offset.z,
                ];
                let pv = Vec3::from(p);
                min = min.min(pv);
                max = max.max(pv);
                Vertex {
                    position: p,
                    normal: v.normal,
                    color: v.color,
                }
            })
            .collect();
        let vertices = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-vertices"),
            contents: bytemuck::cast_slice(&world_verts),
            usage: wgpu::BufferUsages::VERTEX,
        });
        let indices_buf = self.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk-indices"),
            contents: bytemuck::cast_slice(indices),
            usage: wgpu::BufferUsages::INDEX,
        });
        Some(GpuLayer {
            vertices,
            indices: indices_buf,
            index_count: indices.len() as u32,
            aabb_min: min,
            aabb_max: max,
        })
    }

    /// Upload the per-frame globals from the camera and environment.
    pub fn update_globals(&self, camera: &Camera, env: &Environment) {
        let sky = env.sky_color();
        let sun = env.sun_dir();
        let sun_c = env.sun_color();
        let globals = Globals {
            view_proj: camera.view_projection().to_cols_array_2d(),
            camera_pos: [camera.eye.x, camera.eye.y, camera.eye.z, self.render_distance],
            sun_dir: [sun.x, sun.y, sun.z, 0.0],
            sky_color: [sky.x, sky.y, sky.z, 1.0],
            sun_color: [sun_c.x, sun_c.y, sun_c.z, env.ambient()],
        };
        self.queue
            .write_buffer(&self.globals_buffer, 0, bytemuck::bytes_of(&globals));
    }

    /// Encode the opaque + transparent passes into `encoder`, drawing into the
    /// given colour/depth views. `clear` is the sky colour.
    pub fn encode(
        &self,
        encoder: &mut wgpu::CommandEncoder,
        color: &wgpu::TextureView,
        depth: &wgpu::TextureView,
        clear: Vec3,
        frustum: &Frustum,
    ) {
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world-pass"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: color,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color {
                        r: clear.x as f64,
                        g: clear.y as f64,
                        b: clear.z as f64,
                        a: 1.0,
                    }),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Store,
                }),
                stencil_ops: None,
            }),
            timestamp_writes: None,
            occlusion_query_set: None,
        });

        pass.set_pipeline(&self.opaque_pipeline);
        pass.set_bind_group(0, &self.globals_bind_group, &[]);
        for chunk in self.chunks.values() {
            if let Some(layer) = &chunk.opaque {
                if frustum.intersects_aabb(layer.aabb_min, layer.aabb_max) {
                    draw_layer(&mut pass, layer);
                }
            }
        }
        // Ambient critters are opaque; draw them in the same pass.
        if let Some(layer) = &self.entities {
            draw_layer(&mut pass, layer);
        }

        pass.set_pipeline(&self.transparent_pipeline);
        pass.set_bind_group(0, &self.globals_bind_group, &[]);
        for chunk in self.chunks.values() {
            if let Some(layer) = &chunk.transparent {
                if frustum.intersects_aabb(layer.aabb_min, layer.aabb_max) {
                    draw_layer(&mut pass, layer);
                }
            }
        }

        // HUD overlay last, on top of everything (depth-independent).
        if let Some((buf, count)) = &self.ui_buffer {
            pass.set_pipeline(&self.ui_pipeline);
            pass.set_vertex_buffer(0, buf.slice(..));
            pass.draw(0..*count, 0..1);
        }
    }
}

fn draw_layer<'a>(pass: &mut wgpu::RenderPass<'a>, layer: &'a GpuLayer) {
    pass.set_vertex_buffer(0, layer.vertices.slice(..));
    pass.set_index_buffer(layer.indices.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..layer.index_count, 0, 0..1);
}

/// Create a depth texture view sized to the target.
pub fn create_depth(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("depth"),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: DEPTH_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
        view_formats: &[],
    });
    texture.create_view(&wgpu::TextureViewDescriptor::default())
}

fn make_pipeline(
    device: &wgpu::Device,
    layout: &wgpu::PipelineLayout,
    shader: &wgpu::ShaderModule,
    format: wgpu::TextureFormat,
    transparent: bool,
) -> wgpu::RenderPipeline {
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<Vertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 12,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x3,
            },
            wgpu::VertexAttribute {
                offset: 24,
                shader_location: 2,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
    let blend = if transparent {
        Some(wgpu::BlendState::ALPHA_BLENDING)
    } else {
        Some(wgpu::BlendState::REPLACE)
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some(if transparent {
            "transparent-pipeline"
        } else {
            "opaque-pipeline"
        }),
        layout: Some(layout),
        vertex: wgpu::VertexState {
            module: shader,
            entry_point: "vs_main",
            buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend,
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            strip_index_format: None,
            front_face: wgpu::FrontFace::Ccw,
            cull_mode: if transparent {
                None
            } else {
                Some(wgpu::Face::Back)
            },
            unclipped_depth: false,
            polygon_mode: wgpu::PolygonMode::Fill,
            conservative: false,
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: !transparent,
            depth_compare: wgpu::CompareFunction::Less,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Pipeline for the flat 2D HUD overlay: alpha-blended coloured quads in NDC,
/// always drawn on top (depth test Always, no depth write).
fn make_ui_pipeline(
    device: &wgpu::Device,
    format: wgpu::TextureFormat,
) -> wgpu::RenderPipeline {
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ui-shader"),
        source: wgpu::ShaderSource::Wgsl(include_str!("ui.wgsl").into()),
    });
    let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("ui-layout"),
        bind_group_layouts: &[],
        push_constant_ranges: &[],
    });
    let vertex_layout = wgpu::VertexBufferLayout {
        array_stride: std::mem::size_of::<super::ui::UiVertex>() as u64,
        step_mode: wgpu::VertexStepMode::Vertex,
        attributes: &[
            wgpu::VertexAttribute {
                offset: 0,
                shader_location: 0,
                format: wgpu::VertexFormat::Float32x2,
            },
            wgpu::VertexAttribute {
                offset: 8,
                shader_location: 1,
                format: wgpu::VertexFormat::Float32x4,
            },
        ],
    };
    device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
        label: Some("ui-pipeline"),
        layout: Some(&layout),
        vertex: wgpu::VertexState {
            module: &shader,
            entry_point: "vs_main",
            buffers: &[vertex_layout],
            compilation_options: Default::default(),
        },
        fragment: Some(wgpu::FragmentState {
            module: &shader,
            entry_point: "fs_main",
            targets: &[Some(wgpu::ColorTargetState {
                format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            compilation_options: Default::default(),
        }),
        primitive: wgpu::PrimitiveState {
            topology: wgpu::PrimitiveTopology::TriangleList,
            ..Default::default()
        },
        depth_stencil: Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: false,
            depth_compare: wgpu::CompareFunction::Always,
            stencil: wgpu::StencilState::default(),
            bias: wgpu::DepthBiasState::default(),
        }),
        multisample: wgpu::MultisampleState::default(),
        multiview: None,
        cache: None,
    })
}

/// Request an adapter + device. `compatible_surface` is `Some` for the windowed
/// path and `None` for headless offscreen rendering (e.g. lavapipe).
pub async fn request_device(
    instance: &wgpu::Instance,
    compatible_surface: Option<&wgpu::Surface<'_>>,
) -> (wgpu::Adapter, wgpu::Device, wgpu::Queue) {
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            compatible_surface,
            force_fallback_adapter: false,
        })
        .await
        .expect("no suitable GPU adapter");
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("pixelcraft-device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .expect("request device");
    (adapter, device, queue)
}
