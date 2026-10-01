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
use crate::shadows::{Cascades, NEAR_CASCADE, SHADOW_MAP_SIZE, cascades};
use crate::sky::SkyLook;
use crate::textures::{BlockTextures, MIP_LEVELS, TEXTURE_SIZE};
use crate::{ChunkMesh, LodMesh, RenderStats, Scene, Visibility};
use ruda_world::lod::{LOD_TILE_SIZE, LodTilePos};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
/// Bytes of the `Globals` uniform in `world.wgsl`.
const GLOBALS_SIZE: u64 = 464;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
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
    shadows: ShadowMap,
    /// Position of each slot's chunk, as `Rgba32Sint` texels.
    origins: wgpu::Texture,
    origins_view: wgpu::TextureView,
    sky_pipeline: wgpu::RenderPipeline,
    celestial_pipeline: wgpu::RenderPipeline,
    chunk_pipeline: wgpu::RenderPipeline,
    model_pipeline: wgpu::RenderPipeline,
    lod_pipeline: wgpu::RenderPipeline,
    outline_pipeline: wgpu::RenderPipeline,
    crosshair_pipeline: wgpu::RenderPipeline,
    chunks: HashMap<ChunkPos, ChunkEntry>,
    lods: HashMap<LodTilePos, GpuLod>,
    pages: Vec<Page>,
    slots: Slots,
    depth: wgpu::TextureView,
    stats: RenderStats,
    culling: bool,
    /// Scratch space reused every frame.
    visible: Vec<ChunkPos>,
    draws: Vec<Draw>,
    shadow_draws: Vec<Draw>,
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
    /// Everything allocated for the chunk: its quads, then its model
    /// vertices, 16 bytes each.
    space: Range<u32>,
    quads: Range<u32>,
    models: Range<u32>,
    face_counts: [u32; 6],
}

struct Page {
    buffer: wgpu::Buffer,
    space: RangeAllocator,
}

/// Sun shadows: off by default, as they draw the world once more per cascade.
struct ShadowMap {
    enabled: bool,
    /// Sampled by the world pass, a layer per cascade.
    view: wgpu::TextureView,
    /// Drawn into, one per cascade.
    layers: [wgpu::TextureView; 2],
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    /// The globals and chunk positions, without the shadow map itself.
    globals_group: wgpu::BindGroup,
    cascades: [(wgpu::Buffer, wgpu::BindGroup); 2],
}

/// A tile of far-away terrain on the GPU.
struct GpuLod {
    slot: u32,
    page: usize,
    space: Range<u32>,
    /// Lowest to highest point.
    heights: Range<i32>,
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
                wgpu::BindGroupLayoutEntry {
                    binding: 5,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Depth,
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 6,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Comparison),
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
        let shadows = ShadowMap::new(device, &shader, &globals, &origins_view);
        let globals_group = Self::globals_group(
            device,
            &globals_layout,
            &globals,
            &sampler,
            [&block_textures, &origins_view, &sky_textures, &shadows.view],
            &shadows.sampler,
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
        let model_buffer = wgpu::VertexBufferLayout {
            array_stride: QUAD_BYTES,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[wgpu::VertexAttribute {
                format: wgpu::VertexFormat::Uint32x4,
                offset: 0,
                shader_location: 0,
            }],
        };
        let lod_pipeline = pipeline(
            "far terrain",
            &layout,
            "lod_vertex",
            "lod_fragment",
            &[Some(quad_buffer.clone())],
            wgpu::PrimitiveTopology::TriangleStrip,
            Some(wgpu::Face::Back),
            (true, wgpu::CompareFunction::Less),
            None,
        );
        let model_pipeline = pipeline(
            "block models",
            &layout,
            "model_vertex",
            "chunk_fragment",
            &[Some(model_buffer)],
            wgpu::PrimitiveTopology::TriangleList,
            Some(wgpu::Face::Back),
            (true, wgpu::CompareFunction::Less),
            None,
        );
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
            shadows,
            linear_output: color_format.is_srgb(),
            origins,
            origins_view,
            sky_pipeline,
            celestial_pipeline,
            chunk_pipeline,
            model_pipeline,
            lod_pipeline,
            outline_pipeline,
            crosshair_pipeline,
            chunks: HashMap::new(),
            lods: HashMap::new(),
            pages: Vec::new(),
            slots: Slots::new(MAX_SLOTS),
            depth: depth_view(device, width, height),
            stats: RenderStats::default(),
            culling: true,
            visible: Vec::new(),
            draws: Vec::new(),
            shadow_draws: Vec::new(),
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
            [
                &self.block_textures,
                &self.origins_view,
                &self.sky_textures,
                &self.shadows.view,
            ],
            &self.shadows.sampler,
        );
    }

    /// Turns sun shadows on or off.
    pub(crate) fn set_shadows(&mut self, device: &wgpu::Device, enabled: bool) {
        if enabled != self.shadows.enabled {
            self.shadows.set_enabled(device, enabled);
            self.rebuild_globals_group(device);
        }
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
        let gpu = if mesh.is_empty() {
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
        let units: Vec<[u32; 4]> = mesh.quads.iter().chain(&mesh.models).copied().collect();
        let (page, space) = self.store(device, queue, slot, &units);
        self.write_origin(queue, slot, pos.origin().0);
        let quads_end = space.start + mesh.quads.len() as u32;
        Some(GpuChunk {
            slot,
            page,
            quads: space.start..quads_end,
            models: quads_end..space.end,
            space,
            face_counts: mesh.face_counts,
        })
    }

    /// Puts 16-byte units into a page, with the slot in bits 18 to 31 of
    /// their second word, as quads, model vertices and far-away quads all
    /// expect.
    fn store(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        slot: u32,
        units: &[[u32; 4]],
    ) -> (usize, Range<u32>) {
        let len = units.len() as u32;
        let found = self
            .pages
            .iter_mut()
            .enumerate()
            .find_map(|(index, page)| Some((index, page.space.allocate(len)?)));
        let (page, space) = match found {
            Some(found) => found,
            None => {
                let size = PAGE_QUADS.max(len);
                let mut space = RangeAllocator::new(size);
                let units = space.allocate(len).expect("a new page fits the mesh");
                self.pages.push(Page {
                    buffer: device.create_buffer(&wgpu::BufferDescriptor {
                        label: Some("chunk quads"),
                        size: u64::from(size) * QUAD_BYTES,
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                        mapped_at_creation: false,
                    }),
                    space,
                });
                (self.pages.len() - 1, units)
            }
        };
        let bytes: Vec<u8> = units
            .iter()
            .flat_map(|&[a, b, c, d]| [a, b | slot << 18, c, d])
            .flat_map(u32::to_le_bytes)
            .collect();
        queue.write_buffer(
            &self.pages[page].buffer,
            u64::from(space.start) * QUAD_BYTES,
            &bytes,
        );
        (page, space)
    }

    fn write_origin(&self, queue: &wgpu::Queue, slot: u32, origin: glam::IVec3) {
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
    }

    pub(crate) fn upload_lod(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pos: LodTilePos,
        mesh: &LodMesh,
    ) {
        self.remove_lod(pos);
        if mesh.quads.is_empty() {
            return;
        }
        let Some(slot) = self.slots.take() else {
            warn!(?pos, "too many chunks with geometry, not drawing this tile");
            return;
        };
        let (page, space) = self.store(device, queue, slot, &mesh.quads);
        let (x, z) = pos.origin();
        self.write_origin(queue, slot, glam::IVec3::new(x, 0, z));
        self.lods.insert(
            pos,
            GpuLod {
                slot,
                page,
                space,
                heights: mesh.min_y..mesh.max_y,
            },
        );
    }

    pub(crate) fn remove_lod(&mut self, pos: LodTilePos) {
        if let Some(gpu) = self.lods.remove(&pos) {
            self.pages[gpu.page].space.free(gpu.space);
            self.slots.give_back(gpu.slot);
        }
    }

    pub(crate) fn remove(&mut self, pos: ChunkPos) {
        if let Some(gpu) = self.chunks.remove(&pos).and_then(|entry| entry.gpu) {
            self.pages[gpu.page].space.free(gpu.space);
            self.slots.give_back(gpu.slot);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.chunks.clear();
        self.lods.clear();
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
        // With far-away terrain, the view reaches past the chunks drawn in
        // full, and the fog with it.
        let far = scene.view_distance.max(scene.lod_distance);
        let view_proj = scene
            .camera
            .view_proj(width as f32 / height as f32, far + 64.0);
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
        // Far-away terrain gives way to the chunks a chunk inside their edge.
        let lod_start = if scene.lod_distance > 0.0 {
            scene.view_distance - CHUNK_SIZE as f32
        } else {
            f32::MAX
        };
        globals.floats([far * 0.6, far * 0.95, lod_start, 0.0]);
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
        // Shadows come from the sun by day and the moon by night, fading out
        // while either is near the horizon.
        let toward_light = if sky.sun.y >= 0.0 { sky.sun } else { -sky.sun };
        let strength = smoothstep(0.03, 0.25, toward_light.y);
        let shadow_cascades = (self.shadows.enabled && strength > 0.0).then(|| {
            cascades(
                &scene.camera,
                width as f32 / height as f32,
                scene.view_distance,
                toward_light,
            )
        });
        let matrices = shadow_cascades.map_or([glam::Mat4::IDENTITY; 2], |c| c.view_proj);
        for matrix in matrices {
            globals.matrix(matrix);
        }
        let strength = if shadow_cascades.is_some() {
            strength
        } else {
            0.0
        };
        globals.floats(toward_light.extend(strength).to_array());
        globals.floats([
            NEAR_CASCADE,
            shadow_cascades.map_or(0.0, |c| c.end),
            0.0,
            1.0 / SHADOW_MAP_SIZE as f32,
        ]);
        debug_assert_eq!(globals.0.len() as u64, GLOBALS_SIZE);
        queue.write_buffer(&self.globals, 0, &globals.0);
        let clear = wgpu::Color {
            r: f64::from(sky.fog.x),
            g: f64::from(sky.fog.y),
            b: f64::from(sky.fog.z),
            a: 1.0,
        };

        if let Some(cascades) = shadow_cascades {
            self.draw_shadows(queue, encoder, camera, &cascades);
        }

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

        pass.set_pipeline(&self.model_pipeline);
        bound = None;
        for pos in &self.visible {
            let Some(gpu) = self.chunks.get(pos).and_then(|entry| entry.gpu.as_ref()) else {
                continue;
            };
            if gpu.models.is_empty() {
                continue;
            }
            if bound != Some(gpu.page) {
                pass.set_vertex_buffer(0, self.pages[gpu.page].buffer.slice(..));
                bound = Some(gpu.page);
            }
            pass.draw(gpu.models.clone(), 0..1);
        }

        if scene.lod_distance > 0.0 {
            let mut tiles: Vec<&GpuLod> = self
                .lods
                .iter()
                .filter_map(|(pos, gpu)| {
                    let (x, z) = pos.origin();
                    let min =
                        DVec3::new(f64::from(x), f64::from(gpu.heights.start - 1), f64::from(z));
                    let size = DVec3::new(
                        f64::from(LOD_TILE_SIZE),
                        f64::from(gpu.heights.end - gpu.heights.start + 2),
                        f64::from(LOD_TILE_SIZE),
                    );
                    let min = (min - camera).as_vec3();
                    frustum
                        .intersects_box(min, min + size.as_vec3())
                        .then_some(gpu)
                })
                .collect();
            tiles.sort_by_key(|gpu| (gpu.page, gpu.space.start));
            let mut draws = Vec::new();
            for gpu in tiles {
                push_draw(&mut draws, gpu.page, gpu.space.clone());
            }
            self.stats.draw_calls += draws.len();
            pass.set_pipeline(&self.lod_pipeline);
            let mut bound = None;
            for draw in &draws {
                if bound != Some(draw.page) {
                    pass.set_vertex_buffer(0, self.pages[draw.page].buffer.slice(..));
                    bound = Some(draw.page);
                }
                pass.draw(0..4, draw.quads.clone());
            }
        }

        if scene.target.is_some() {
            pass.set_pipeline(&self.outline_pipeline);
            pass.draw(0..24, 0..1);
        }
        pass.set_pipeline(&self.crosshair_pipeline);
        pass.draw(0..12, 0..1);
    }

    /// Draws every chunk the light sees into each cascade of the shadow map.
    /// Cave culling doesn't apply: what the camera can't see may still cast
    /// a shadow into view.
    fn draw_shadows(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: DVec3,
        cascades: &Cascades,
    ) {
        for (cascade, view_proj) in cascades.view_proj.iter().enumerate() {
            let (buffer, group) = &self.shadows.cascades[cascade];
            let mut matrix = Std140::default();
            matrix.matrix(*view_proj);
            queue.write_buffer(buffer, 0, &matrix.0);
            let frustum = Frustum::new(*view_proj);
            let mut casters: Vec<&GpuChunk> = self
                .chunks
                .iter()
                .filter_map(|(pos, entry)| {
                    let min = (pos.origin().0.as_dvec3() - camera).as_vec3();
                    let seen = frustum.intersects_box(min, min + Vec3::splat(CHUNK_SIZE as f32));
                    entry.gpu.as_ref().filter(|_| seen)
                })
                .collect();
            // In buffer order, so neighbours merge into one draw.
            casters.sort_by_key(|gpu| (gpu.page, gpu.quads.start));
            self.shadow_draws.clear();
            for gpu in casters {
                push_draw(&mut self.shadow_draws, gpu.page, gpu.quads.clone());
            }
            self.stats.shadow_draw_calls += self.shadow_draws.len();

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow map"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadows.layers[cascade],
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.shadows.pipeline);
            pass.set_bind_group(0, &self.shadows.globals_group, &[]);
            pass.set_bind_group(1, group, &[]);
            let mut bound = None;
            for draw in &self.shadow_draws {
                if bound != Some(draw.page) {
                    pass.set_vertex_buffer(0, self.pages[draw.page].buffer.slice(..));
                    bound = Some(draw.page);
                }
                pass.draw(0..4, draw.quads.clone());
            }
        }
    }

    fn globals_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        sampler: &wgpu::Sampler,
        [textures, origins, sky, shadows]: [&wgpu::TextureView; 4],
        shadow_sampler: &wgpu::Sampler,
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
                wgpu::BindGroupEntry {
                    binding: 5,
                    resource: wgpu::BindingResource::TextureView(shadows),
                },
                wgpu::BindGroupEntry {
                    binding: 6,
                    resource: wgpu::BindingResource::Sampler(shadow_sampler),
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

impl ShadowMap {
    fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        globals: &wgpu::Buffer,
        origins: &wgpu::TextureView,
    ) -> Self {
        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow map globals"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: wgpu::BufferSize::new(GLOBALS_SIZE),
                    },
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
            ],
        });
        let cascade_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("shadow cascade"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: wgpu::BufferSize::new(64),
                },
                count: None,
            }],
        });
        let globals_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("shadow map globals"),
            layout: &globals_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: globals.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: wgpu::BindingResource::TextureView(origins),
                },
            ],
        });
        let cascades = [0, 1].map(|_| {
            let buffer = device.create_buffer(&wgpu::BufferDescriptor {
                label: Some("shadow cascade"),
                size: 64,
                usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
                mapped_at_creation: false,
            });
            let group = device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("shadow cascade"),
                layout: &cascade_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: buffer.as_entire_binding(),
                }],
            });
            (buffer, group)
        });
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("shadow map"),
            bind_group_layouts: &[Some(&globals_layout), Some(&cascade_layout)],
            immediate_size: 0,
        });
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("shadow map"),
            layout: Some(&layout),
            vertex: wgpu::VertexState {
                module: shader,
                entry_point: Some("shadow_vertex"),
                compilation_options: Default::default(),
                buffers: &[Some(wgpu::VertexBufferLayout {
                    array_stride: QUAD_BYTES,
                    step_mode: wgpu::VertexStepMode::Instance,
                    attributes: &[wgpu::VertexAttribute {
                        format: wgpu::VertexFormat::Uint32x4,
                        offset: 0,
                        shader_location: 0,
                    }],
                })],
            },
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::TriangleStrip,
                ..Default::default()
            },
            depth_stencil: Some(wgpu::DepthStencilState {
                format: SHADOW_FORMAT,
                depth_write_enabled: Some(true),
                depth_compare: Some(wgpu::CompareFunction::LessEqual),
                stencil: Default::default(),
                // Keeps lit faces from shading themselves.
                bias: wgpu::DepthBiasState {
                    constant: 2,
                    slope_scale: 2.0,
                    clamp: 0.0,
                },
            }),
            multisample: Default::default(),
            fragment: None,
            multiview_mask: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            label: Some("shadow map"),
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            compare: Some(wgpu::CompareFunction::LessEqual),
            ..Default::default()
        });
        let (view, layers) = shadow_texture(device, 1);
        Self {
            enabled: false,
            view,
            layers,
            sampler,
            pipeline,
            globals_group,
            cascades,
        }
    }

    /// Off, the map shrinks to a texel per cascade.
    fn set_enabled(&mut self, device: &wgpu::Device, enabled: bool) {
        let size = if enabled { SHADOW_MAP_SIZE } else { 1 };
        (self.view, self.layers) = shadow_texture(device, size);
        self.enabled = enabled;
    }
}

/// A depth texture with a layer per cascade, as one view to sample and one
/// per layer to draw into.
fn shadow_texture(device: &wgpu::Device, size: u32) -> (wgpu::TextureView, [wgpu::TextureView; 2]) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow map"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 2,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: SHADOW_FORMAT,
        usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
        view_formats: &[],
    });
    let view = texture.create_view(&wgpu::TextureViewDescriptor {
        dimension: Some(wgpu::TextureViewDimension::D2Array),
        ..Default::default()
    });
    let layers = [0, 1].map(|layer| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer,
            array_layer_count: Some(1),
            ..Default::default()
        })
    });
    (view, layers)
}

fn smoothstep(edge0: f32, edge1: f32, x: f32) -> f32 {
    let t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
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
