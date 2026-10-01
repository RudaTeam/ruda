//! Draws the block world, the outline of the targeted block and the crosshair.

use std::collections::HashMap;

use glam::Vec3;
use ruda_core::{CHUNK_SIZE, ChunkPos};
use wgpu::util::DeviceExt as _;

use crate::camera::Frustum;
use crate::textures::{BlockTextures, MIP_LEVELS, TEXTURE_SIZE};
use crate::{ChunkMesh, Scene};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
/// Bytes of the `Globals` uniform in `world.wgsl`.
const GLOBALS_SIZE: u64 = 128;
/// Bytes of the `Chunk` uniform in `world.wgsl`.
const CHUNK_UNIFORM_SIZE: u64 = 16;

pub(crate) struct WorldPass {
    globals: wgpu::Buffer,
    globals_layout: wgpu::BindGroupLayout,
    globals_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    /// One slot per chunk drawn this frame, addressed with dynamic offsets.
    chunk_uniforms: wgpu::Buffer,
    chunk_layout: wgpu::BindGroupLayout,
    chunk_group: wgpu::BindGroup,
    chunk_slots: u64,
    slot_size: u64,
    chunk_pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    crosshair_pipeline: wgpu::RenderPipeline,
    meshes: HashMap<ChunkPos, GpuMesh>,
    depth: wgpu::TextureView,
}

struct GpuMesh {
    buffer: wgpu::Buffer,
    quads: u32,
}

impl WorldPass {
    pub(crate) fn new(
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        color_format: wgpu::TextureFormat,
        width: u32,
        height: u32,
    ) -> Self {
        let shader = device.create_shader_module(wgpu::include_wgsl!("world.wgsl"));

        let globals = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("world globals"),
            size: GLOBALS_SIZE,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("world globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(GLOBALS_SIZE),
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("block textures"),
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Nearest,
            min_filter: wgpu::FilterMode::Nearest,
            mipmap_filter: wgpu::MipmapFilterMode::Linear,
            ..Default::default()
        });
        let textures = BlockTextures::load(&ruda_core::ContentBuilder::new().build(), 2);
        let globals_group = Self::globals_group(
            device,
            &globals_layout,
            &globals,
            &sampler,
            &upload_textures(device, queue, &textures),
        );

        let chunk_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("chunk"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(CHUNK_UNIFORM_SIZE),
                },
                count: None,
            }],
        });
        let slot_size =
            u64::from(device.limits().min_uniform_buffer_offset_alignment).max(CHUNK_UNIFORM_SIZE);
        let chunk_slots = 256;
        let (chunk_uniforms, chunk_group) =
            Self::chunk_uniforms(device, &chunk_layout, chunk_slots, slot_size);

        let chunk_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("chunks"),
                bind_group_layouts: &[Some(&globals_layout), Some(&chunk_layout)],
                immediate_size: 0,
            });
        let overlay_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("overlay"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let pipeline = |label,
                        layout,
                        vertex,
                        fragment,
                        buffers: &[Option<wgpu::VertexBufferLayout>],
                        topology,
                        cull_mode,
                        depth: (bool, wgpu::CompareFunction)| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers,
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    front_face: wgpu::FrontFace::Ccw,
                    cull_mode,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: DEPTH_FORMAT,
                    depth_write_enabled: Some(depth.0),
                    depth_compare: Some(depth.1),
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets: &[Some(color_format.into())],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let quad_buffer = wgpu::VertexBufferLayout {
            array_stride: 8,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32x2,
                offset: 0,
                shader_location: 0,
            }],
        };
        let chunk_pipeline = pipeline(
            "chunks",
            &chunk_pipeline_layout,
            "chunk_vertex",
            "chunk_fragment",
            &[Some(quad_buffer)],
            wgpu::PrimitiveTopology::TriangleStrip,
            Some(wgpu::Face::Back),
            (true, wgpu::CompareFunction::Less),
        );
        let outline_pipeline = pipeline(
            "block outline",
            &overlay_layout,
            "outline_vertex",
            "outline_fragment",
            &[],
            wgpu::PrimitiveTopology::LineList,
            None,
            (false, wgpu::CompareFunction::LessEqual),
        );
        let crosshair_pipeline = pipeline(
            "crosshair",
            &overlay_layout,
            "crosshair_vertex",
            "crosshair_fragment",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            None,
            (false, wgpu::CompareFunction::Always),
        );

        Self {
            globals,
            globals_layout,
            globals_group,
            sampler,
            chunk_uniforms,
            chunk_layout,
            chunk_group,
            chunk_slots,
            slot_size,
            chunk_pipeline,
            outline_pipeline,
            crosshair_pipeline,
            meshes: HashMap::new(),
            depth: depth_view(device, width, height),
        }
    }

    pub(crate) fn set_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &BlockTextures,
    ) {
        let view = upload_textures(device, queue, textures);
        self.globals_group = Self::globals_group(
            device,
            &self.globals_layout,
            &self.globals,
            &self.sampler,
            &view,
        );
    }

    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth = depth_view(device, width, height);
    }

    pub(crate) fn upload(&mut self, device: &wgpu::Device, pos: ChunkPos, mesh: &ChunkMesh) {
        if mesh.is_empty() {
            self.meshes.remove(&pos);
            return;
        }
        let bytes: Vec<u8> = mesh
            .quads
            .iter()
            .flatten()
            .flat_map(|w| w.to_le_bytes())
            .collect();
        let buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("chunk mesh"),
            contents: &bytes,
            usage: wgpu::BufferUsages::VERTEX,
        });
        self.meshes.insert(
            pos,
            GpuMesh {
                buffer,
                quads: mesh.quads.len() as u32,
            },
        );
    }

    pub(crate) fn remove(&mut self, pos: ChunkPos) {
        self.meshes.remove(&pos);
    }

    pub(crate) fn chunk_count(&self) -> usize {
        self.meshes.len()
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        scene: &Scene,
        (width, height): (u32, u32),
        sky: wgpu::Color,
    ) {
        let camera = scene.camera.position;
        let view_proj = scene
            .camera
            .view_proj(width as f32 / height as f32, scene.view_distance + 64.0);
        let frustum = Frustum::new(view_proj);
        let size = CHUNK_SIZE as f32;

        // Visible chunks, nearest first so the depth test rejects more.
        let mut visible: Vec<(ChunkPos, Vec3)> = self
            .meshes
            .keys()
            .map(|&pos| (pos, (pos.origin().0.as_dvec3() - camera).as_vec3()))
            .filter(|(_, origin)| frustum.intersects_box(*origin, *origin + size))
            .collect();
        visible.sort_by(|a, b| a.1.length_squared().total_cmp(&b.1.length_squared()));

        if visible.len() as u64 > self.chunk_slots {
            self.chunk_slots = (visible.len() as u64).next_power_of_two();
            (self.chunk_uniforms, self.chunk_group) =
                Self::chunk_uniforms(device, &self.chunk_layout, self.chunk_slots, self.slot_size);
        }
        if !visible.is_empty() {
            let mut origins = vec![0u8; visible.len() * self.slot_size as usize];
            for (slot, (_, origin)) in origins
                .chunks_exact_mut(self.slot_size as usize)
                .zip(&visible)
            {
                write_floats(&mut slot[..16], &[origin.x, origin.y, origin.z, 0.0]);
            }
            queue.write_buffer(&self.chunk_uniforms, 0, &origins);
        }

        let selection = scene
            .target
            .map(|block| (block.0.as_dvec3() - camera).as_vec3());
        let mut globals = [0u8; GLOBALS_SIZE as usize];
        write_floats(&mut globals[..64], &view_proj.to_cols_array());
        write_floats(
            &mut globals[64..],
            &[
                sky.r as f32,
                sky.g as f32,
                sky.b as f32,
                1.0,
                scene.view_distance * 0.6,
                scene.view_distance * 0.95,
                0.0,
                0.0,
            ],
        );
        let selection = selection.unwrap_or(Vec3::ZERO);
        write_floats(
            &mut globals[96..],
            &[
                selection.x,
                selection.y,
                selection.z,
                0.0,
                width as f32,
                height as f32,
                0.0,
                0.0,
            ],
        );
        queue.write_buffer(&self.globals, 0, &globals);

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(sky),
                    store: wgpu::StoreOp::Store,
                },
            })],
            depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                view: &self.depth,
                depth_ops: Some(wgpu::Operations {
                    load: wgpu::LoadOp::Clear(1.0),
                    store: wgpu::StoreOp::Discard,
                }),
                stencil_ops: None,
            }),
            ..Default::default()
        });
        pass.set_bind_group(0, &self.globals_group, &[]);

        pass.set_pipeline(&self.chunk_pipeline);
        for (slot, (pos, _)) in visible.iter().enumerate() {
            let mesh = &self.meshes[pos];
            let offset = (slot as u64 * self.slot_size) as u32;
            pass.set_bind_group(1, &self.chunk_group, &[offset]);
            pass.set_vertex_buffer(0, mesh.buffer.slice(..));
            pass.draw(0..4, 0..mesh.quads);
        }

        if scene.target.is_some() {
            pass.set_pipeline(&self.outline_pipeline);
            pass.draw(0..24, 0..1);
        }
        pass.set_pipeline(&self.crosshair_pipeline);
        pass.draw(0..12, 0..1);
    }

    fn globals_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        textures: &wgpu::TextureView,
    ) -> wgpu::BindGroup {
        device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("world globals"),
            layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::TextureView(textures),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::Sampler(sampler),
                },
            ],
        })
    }

    fn chunk_uniforms(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        slots: u64,
        slot_size: u64,
    ) -> (wgpu::Buffer, wgpu::BindGroup) {
        let buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("chunk origins"),
            size: slots * slot_size,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("chunk origins"),
            layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(CHUNK_UNIFORM_SIZE),
                }),
            }],
        });
        (buffer, group)
    }
}

fn upload_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    textures: &BlockTextures,
) -> wgpu::TextureView {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("block textures"),
        size: wgpu::Extent3d {
            width: TEXTURE_SIZE,
            height: TEXTURE_SIZE,
            depth_or_array_layers: textures.layers,
        },
        mip_level_count: MIP_LEVELS,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (level, pixels) in textures.mips.iter().enumerate() {
        let size = TEXTURE_SIZE >> level;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(size),
            },
            wgpu::Extent3d {
                width: size,
                height: size,
                depth_or_array_layers: textures.layers,
            },
        );
    }
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
}

fn depth_view(device: &wgpu::Device, width: u32, height: u32) -> wgpu::TextureView {
    device
        .create_texture(&wgpu::TextureDescriptor {
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
        })
        .create_view(&Default::default())
}

fn write_floats(out: &mut [u8], values: &[f32]) {
    for (bytes, value) in out.as_chunks_mut::<4>().0.iter_mut().zip(values) {
        *bytes = value.to_le_bytes();
    }
}

#[cfg(test)]
mod tests {
    /// Catches shader mistakes without a GPU, with the same compiler wgpu uses.
    #[test]
    fn shader_is_valid_on_the_baseline_tier() {
        let module = naga::front::wgsl::parse_str(include_str!("world.wgsl")).unwrap();
        naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
    }
}
