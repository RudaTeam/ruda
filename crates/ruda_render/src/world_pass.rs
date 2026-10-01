//! Draws the block world, the outline of the targeted block and the crosshair.
//!
//! Chunk geometry lives in a few large shared buffers ("pages"). Every chunk
//! gets a slot whose texel in a small texture holds the chunk's position, and
//! its quads carry the slot, so drawing needs no per-chunk state: each draw
//! is just a range of quads, and neighbouring ranges merge into one draw.

use std::collections::HashMap;
use std::ops::Range;

use glam::{DVec3, Vec3};
use ruda_core::{CHUNK_SIZE, ChunkPos, Face, WorldBounds};
use tracing::warn;

use crate::arena::{RangeAllocator, Slots};
use crate::camera::Frustum;
use crate::culling::visible_chunks;
use crate::sky::SkyLook;
use crate::textures::{BlockTextures, MIP_LEVELS, TEXTURE_SIZE};
use crate::{ChunkMesh, RenderStats, Scene, Visibility};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
/// Bytes of the `Globals` uniform in `world.wgsl`.
const GLOBALS_SIZE: u64 = 304;
/// Edge length of the sun and moon images; smaller ones are centred.
const SKY_TEXTURE_SIZE: u32 = 32;
const QUAD_BYTES: u64 = 16;
/// Quads per page: 8 MiB.
const PAGE_QUADS: u32 = 1 << 19;
/// Width and height of the chunk position texture, `ORIGINS_WIDTH` in
/// `world.wgsl`. Its texels are the chunk slots.
const ORIGINS_WIDTH: u32 = 128;
const MAX_SLOTS: u32 = ORIGINS_WIDTH * ORIGINS_WIDTH;

pub(crate) struct WorldPass {
    globals: wgpu::Buffer,
    globals_layout: wgpu::BindGroupLayout,
    globals_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    block_textures: wgpu::TextureView,
    /// The sun and the moon.
    sky_textures: wgpu::TextureView,
    /// Whether the target takes linear colours and encodes them as sRGB.
    linear_output: bool,
    /// Position of each slot's chunk, as `Rgba32Sint` texels.
    origins: wgpu::Texture,
    origins_view: wgpu::TextureView,
    sky_pipeline: wgpu::RenderPipeline,
    celestial_pipeline: wgpu::RenderPipeline,
    chunk_pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    crosshair_pipeline: wgpu::RenderPipeline,
    chunks: HashMap<ChunkPos, ChunkEntry>,
    pages: Vec<Page>,
    slots: Slots,
    depth: wgpu::TextureView,
    stats: RenderStats,
    culling: bool,
    /// Scratch space reused every frame.
    visible: Vec<ChunkPos>,
    draws: Vec<Draw>,
}

/// What the renderer knows about a loaded chunk.
struct ChunkEntry {
    visibility: Visibility,
    /// `None` for chunks without geometry, like air.
    gpu: Option<GpuChunk>,
}

struct GpuChunk {
    slot: u32,
    page: usize,
    quads: Range<u32>,
    face_counts: [u32; 6],
}

struct Page {
    buffer: wgpu::Buffer,
    space: RangeAllocator,
}

/// A run of quads in one page.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Draw {
    page: usize,
    quads: Range<u32>,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Sint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 4,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
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
        let block_textures = upload_textures(device, queue, &textures);
        let origins = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("chunk origins"),
            size: wgpu::Extent3d {
                width: ORIGINS_WIDTH,
                height: ORIGINS_WIDTH,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba32Sint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let origins_view = origins.create_view(&Default::default());
        let sky_textures = upload_sky_textures(device, queue, [None, None]);
        let globals_group = Self::globals_group(
            device,
            &globals_layout,
            &globals,
            &sampler,
            [&block_textures, &origins_view, &sky_textures],
        );

        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world"),
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
                        depth: (bool, wgpu::CompareFunction),
                        blend: Option<wgpu::BlendState>| {
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
                    targets: &[Some(wgpu::ColorTargetState {
                        format: color_format,
                        blend,
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        let quad_buffer = wgpu::VertexBufferLayout {
            array_stride: QUAD_BYTES,
            step_mode: wgpu::VertexStepMode::Instance,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32x4,
                offset: 0,
                shader_location: 0,
            }],
        };
        let chunk_pipeline = pipeline(
            "chunks",
            &layout,
            "chunk_vertex",
            "chunk_fragment",
            &[Some(quad_buffer)],
            wgpu::PrimitiveTopology::TriangleStrip,
            Some(wgpu::Face::Back),
            (true, wgpu::CompareFunction::Less),
            None,
        );
        let sky_pipeline = pipeline(
            "sky",
            &layout,
            "sky_vertex",
            "sky_fragment",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            None,
            (false, wgpu::CompareFunction::Always),
            None,
        );
        let celestial_pipeline = pipeline(
            "sun and moon",
            &layout,
            "celestial_vertex",
            "celestial_fragment",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            None,
            (false, wgpu::CompareFunction::Always),
            Some(wgpu::BlendState::ALPHA_BLENDING),
        );
        let outline_pipeline = pipeline(
            "block outline",
            &layout,
            "outline_vertex",
            "outline_fragment",
            &[],
            wgpu::PrimitiveTopology::LineList,
            None,
            (false, wgpu::CompareFunction::LessEqual),
            None,
        );
        let crosshair_pipeline = pipeline(
            "crosshair",
            &layout,
            "crosshair_vertex",
            "crosshair_fragment",
            &[],
            wgpu::PrimitiveTopology::TriangleList,
            None,
            (false, wgpu::CompareFunction::Always),
            None,
        );

        Self {
            globals,
            globals_layout,
            globals_group,
            sampler,
            block_textures,
            sky_textures,
            linear_output: color_format.is_srgb(),
            origins,
            origins_view,
            sky_pipeline,
            celestial_pipeline,
            chunk_pipeline,
            outline_pipeline,
            crosshair_pipeline,
            chunks: HashMap::new(),
            pages: Vec::new(),
            slots: Slots::new(MAX_SLOTS),
            depth: depth_view(device, width, height),
            stats: RenderStats::default(),
            culling: true,
            visible: Vec::new(),
            draws: Vec::new(),
        }
    }

    pub(crate) fn set_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        textures: &BlockTextures,
    ) {
        self.block_textures = upload_textures(device, queue, textures);
        self.rebuild_globals_group(device);
    }

    /// The sun and moon images, PNG-encoded, at most 32 pixels square.
    pub(crate) fn set_sky_textures(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        sun: &[u8],
        moon: &[u8],
    ) {
        let decode = |png: &[u8], name: &str| {
            crate::textures::decode_image(png, SKY_TEXTURE_SIZE)
                .inspect_err(|error| warn!("broken {name} image: {error:#}"))
                .ok()
        };
        self.sky_textures =
            upload_sky_textures(device, queue, [decode(sun, "sun"), decode(moon, "moon")]);
        self.rebuild_globals_group(device);
    }

    fn rebuild_globals_group(&mut self, device: &wgpu::Device) {
        self.globals_group = Self::globals_group(
            device,
            &self.globals_layout,
            &self.globals,
            &self.sampler,
            [&self.block_textures, &self.origins_view, &self.sky_textures],
        );
    }

    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth = depth_view(device, width, height);
    }

    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pos: ChunkPos,
        mesh: &ChunkMesh,
    ) {
        self.remove(pos);
        let gpu = if mesh.quads.is_empty() {
            None
        } else {
            self.place(device, queue, pos, mesh)
        };
        self.chunks.insert(
            pos,
            ChunkEntry {
                visibility: mesh.visibility,
                gpu,
            },
        );
    }

    /// Finds room for a chunk's quads and its slot, and uploads them.
    fn place(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pos: ChunkPos,
        mesh: &ChunkMesh,
    ) -> Option<GpuChunk> {
        let Some(slot) = self.slots.take() else {
            warn!(?pos, "too many chunks with geometry, not drawing this one");
            return None;
        };
        let len = mesh.quads.len() as u32;
        let found = self
            .pages
            .iter_mut()
            .enumerate()
            .find_map(|(index, page)| Some((index, page.space.allocate(len)?)));
        let (page, quads) = match found {
            Some(found) => found,
            None => {
                let size = PAGE_QUADS.max(len);
                let mut space = RangeAllocator::new(size);
                let quads = space.allocate(len).expect("a new page fits the mesh");
                self.pages.push(Page {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("chunk quads"),
                        size: u64::from(size) * QUAD_BYTES,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    space,
                });
                (self.pages.len() - 1, quads)
            }
        };

        let bytes: Vec<u8> = mesh
            .quads
            .iter()
            .flat_map(|&[shape, look, light_a, light_b]| {
                [shape, look | slot << 18, light_a, light_b]
            })
            .flat_map(u32::to_le_bytes)
            .collect();
        queue.write_buffer(
            &self.pages[page].buffer,
            u64::from(quads.start) * QUAD_BYTES,
            &bytes,
        );
        let origin = pos.origin().0;
        let texel: Vec<u8> = [origin.x, origin.y, origin.z, 0]
            .into_iter()
            .flat_map(i32::to_le_bytes)
            .collect();
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &self.origins,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: slot % ORIGINS_WIDTH,
                    y: slot / ORIGINS_WIDTH,
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &texel,
            wgpu::TexelCopyBufferLayout::default(),
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Some(GpuChunk {
            slot,
            page,
            quads,
            face_counts: mesh.face_counts,
        })
    }

    pub(crate) fn remove(&mut self, pos: ChunkPos) {
        if let Some(gpu) = self.chunks.remove(&pos).and_then(|entry| entry.gpu) {
            self.pages[gpu.page].space.free(gpu.quads);
            self.slots.give_back(gpu.slot);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.chunks.clear();
        self.pages.clear();
        self.slots = Slots::new(MAX_SLOTS);
    }

    /// Chunks with geometry on the GPU.
    pub(crate) fn chunk_count(&self) -> usize {
        self.chunks
            .values()
            .filter(|entry| entry.gpu.is_some())
            .count()
    }

    pub(crate) fn stats(&self) -> RenderStats {
        self.stats
    }

    pub(crate) fn set_culling(&mut self, culling: bool) {
        self.culling = culling;
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn draw(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::TextureView,
        scene: &Scene,
        (width, height): (u32, u32),
    ) {
        let camera = scene.camera.position;
        let view_proj = scene
            .camera
            .view_proj(width as f32 / height as f32, scene.view_distance + 64.0);
        let frustum = Frustum::new(view_proj);
        let radius = (scene.view_distance / CHUNK_SIZE as f32).ceil() as i32;
        let bounds = scene.bounds.unwrap_or(WorldBounds::DEFAULT);
        let world_y = (bounds.min_y >> 5)..=(bounds.max_y >> 5);
        let chunks = &self.chunks;
        if self.culling {
            visible_chunks(
                camera,
                &frustum,
                radius,
                world_y,
                |pos| chunks.get(&pos).map(|entry| entry.visibility),
                &mut self.visible,
            );
        } else {
            self.visible.clear();
            self.visible.extend(chunks.keys().copied().filter(|pos| {
                let min = (pos.origin().0.as_dvec3() - camera).as_vec3();
                frustum.intersects_box(min, min + Vec3::splat(CHUNK_SIZE as f32))
            }));
        }

        self.draws.clear();
        self.stats = RenderStats {
            chunks: self.chunk_count(),
            ..Default::default()
        };
        for pos in &self.visible {
            let Some(gpu) = self.chunks.get(pos).and_then(|entry| entry.gpu.as_ref()) else {
                continue;
            };
            self.stats.drawn_chunks += 1;
            let facing = if self.culling {
                facing_camera(*pos, camera)
            } else {
                [true; 6]
            };
            let mut start = gpu.quads.start;
            for face in Face::ALL {
                let count = gpu.face_counts[face.index()];
                if count > 0 && facing[face.index()] {
                    push_draw(&mut self.draws, gpu.page, start..start + count);
                    self.stats.quads += u64::from(count);
                }
                start += count;
            }
        }
        self.stats.draw_calls = self.draws.len();

        let selection = scene
            .target
            .map(|block| (block.0.as_dvec3() - camera).as_vec3());
        let camera_block = camera.floor();
        let camera_fract = (camera - camera_block).as_vec3();
        let camera_block = camera_block.as_ivec3();
        let mut sky = SkyLook::at(scene.time_of_day, scene.eye_light);
        if self.linear_output {
            sky = sky.to_linear();
        }
        let selection = selection.unwrap_or(Vec3::ZERO);
        let mut globals = Std140::default();
        globals.matrix(view_proj);
        globals.matrix(view_proj.inverse());
        globals.floats([sky.fog.x, sky.fog.y, sky.fog.z, 1.0]);
        globals.floats([
            scene.view_distance * 0.6,
            scene.view_distance * 0.95,
            0.0,
            0.0,
        ]);
        globals.floats(selection.extend(0.0).to_array());
        globals.floats([width as f32, height as f32, 0.0, 0.0]);
        globals.ints(camera_block.extend(0).to_array());
        globals.floats(camera_fract.extend(0.0).to_array());
        globals.floats(sky.sky_light.extend(sky.ambient).to_array());
        globals.floats(sky.sun.extend(sky.day).to_array());
        globals.floats(sky.zenith.extend(sky.stars).to_array());
        // The horizon's w is how visible the sun and moon are: not from caves.
        let eye = scene.eye_light.sky() as f32 / 15.0;
        globals.floats(sky.horizon.extend(eye).to_array());
        globals.floats(sky.glow.extend(0.0).to_array());
        debug_assert_eq!(globals.0.len() as u64, GLOBALS_SIZE);
        queue.write_buffer(&self.globals, 0, &globals.0);
        let clear = wgpu::Color {
            r: f64::from(sky.fog.x),
            g: f64::from(sky.fog.y),
            b: f64::from(sky.fog.z),
            a: 1.0,
        };

        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(clear),
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

        pass.set_pipeline(&self.sky_pipeline);
        pass.draw(0..3, 0..1);
        pass.set_pipeline(&self.celestial_pipeline);
        pass.draw(0..12, 0..1);

        pass.set_pipeline(&self.chunk_pipeline);
        let mut bound = None;
        for draw in &self.draws {
            if bound != Some(draw.page) {
                pass.set_vertex_buffer(0, self.pages[draw.page].buffer.slice(..));
                bound = Some(draw.page);
            }
            pass.draw(0..4, draw.quads.clone());
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
        [textures, origins, sky]: [&wgpu::TextureView; 3],
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
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(origins),
                },
                wgpu::BindGroupEntry {
                    binding: 4,
                    resource: wgpu::BindingResource::TextureView(sky),
                },
            ],
        })
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

/// Which way the quads of a chunk may face the camera, in [`Face::ALL`] order.
fn facing_camera(pos: ChunkPos, camera: DVec3) -> [bool; 6] {
    let min = pos.origin().0.as_dvec3();
    let max = min + f64::from(CHUNK_SIZE);
    Face::ALL.map(|face| {
        let axis = face.axis();
        if face.is_positive() {
            camera[axis] > min[axis]
        } else {
            camera[axis] < max[axis]
        }
    })
}

/// Adds a range of quads, extending the last draw if it ends where the range
/// starts.
fn push_draw(draws: &mut Vec<Draw>, page: usize, quads: Range<u32>) {
    if let Some(last) = draws.last_mut()
        && last.page == page
        && last.quads.end == quads.start
    {
        last.quads.end = quads.end;
        return;
    }
    draws.push(Draw { page, quads });
}

/// Uniform data laid out in 16-byte rows, as WGSL expects it.
#[derive(Default)]
struct Std140(Vec<u8>);

impl Std140 {
    fn floats(&mut self, values: [f32; 4]) {
        self.0.extend(values.into_iter().flat_map(f32::to_le_bytes));
    }

    fn ints(&mut self, values: [i32; 4]) {
        self.0.extend(values.into_iter().flat_map(i32::to_le_bytes));
    }

    fn matrix(&mut self, matrix: glam::Mat4) {
        for column in matrix.to_cols_array_2d() {
            self.floats(column);
        }
    }
}

/// The sun and moon as two layers of a texture array; missing ones are
/// left transparent.
fn upload_sky_textures(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    images: [Option<(u32, u32, Vec<u8>)>; 2],
) -> wgpu::TextureView {
    let size = SKY_TEXTURE_SIZE;
    let mut pixels = vec![0u8; (size * size * 4 * 2) as usize];
    for (layer, image) in images.iter().enumerate() {
        let Some((width, height, rgba)) = image else {
            continue;
        };
        let (left, top) = ((size - width) / 2, (size - height) / 2);
        for y in 0..*height {
            for x in 0..*width {
                let from = ((y * width + x) * 4) as usize;
                let to = (((layer as u32 * size + top + y) * size + left + x) * 4) as usize;
                pixels[to..to + 4].copy_from_slice(&rgba[from..from + 4]);
            }
        }
    }
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sun and moon"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 2,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    queue.write_texture(
        texture.as_image_copy(),
        &pixels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(size * 4),
            rows_per_image: Some(size),
        },
        wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 2,
        },
    );
    texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    })
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
