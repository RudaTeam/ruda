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
use crate::atmosphere::{SKY_AZIMUTHS, SKY_ELEVATIONS, SKY_LIGHTS, SkyTables};
use crate::camera::Frustum;
use crate::clouds::{
    CELL_MAP_PERIOD, CELL_MAP_SIZE, CLOUD_BOTTOM, CLOUD_TOP, ColumnTops, OBSTACLE_CELL,
    OBSTACLE_SIZE, ObstacleMap, cloud_cells, cloud_depths,
};
use crate::culling::visible_chunks;
use crate::gpu_timer::GpuTimer;
use crate::rays::{RayViews, RayWorld};
use crate::shadows::{CASCADES, SHADOW_MAP_SIZE, cascade, margin, reach};
use crate::sky::{Eye, SkyLook};
use crate::textures::{BlockTextures, MIP_LEVELS, TEXTURE_SIZE};
use crate::{ChunkMesh, CloudSky, LodMesh, RenderStats, Scene, Visibility};
use ruda_world::lod::{LOD_TILE_SIZE, LodTilePos};

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Bytes of the `Globals` uniform in `world.wgsl`.
const GLOBALS_SIZE: u64 = 832;
const SHADOW_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth32Float;
/// Haze per block at sea level; it thins out with height (see `world.wgsl`).
const HAZE: f32 = 4e-4;
/// Edge length of the sun and moon images; smaller ones are centred.
const SKY_TEXTURE_SIZE: u32 = 32;
const QUAD_BYTES: u64 = 16;
/// Quads per page: 8 MiB.
const PAGE_QUADS: u32 = 1 << 19;
/// Width and height of the chunk position texture, `ORIGINS_WIDTH` in
/// `world.wgsl`. Its texels are the chunk slots.
const ORIGINS_WIDTH: u32 = 128;
const MAX_SLOTS: u32 = ORIGINS_WIDTH * ORIGINS_WIDTH;
/// Width and height of the map of chunks drawn in full, `DRAWN_WIDTH` in
/// `world.wgsl`: a texel per column of chunks, repeating, so it must be
/// more than twice as wide as the farthest view distance.
const DRAWN_WIDTH: u32 = 128;

pub(crate) struct WorldPass {
    globals: wgpu::Buffer,
    globals_layout: wgpu::BindGroupLayout,
    globals_group: wgpu::BindGroup,
    sampler: wgpu::Sampler,
    block_textures: wgpu::TextureView,
    /// The sun and the moon.
    sky_textures: wgpu::TextureView,
    /// The sky's colour by direction, per height of the light.
    sky_table: wgpu::TextureView,
    /// Whether the target takes linear colours and encodes them as sRGB.
    linear_output: bool,
    lighting: Lighting,
    shadows: ShadowMap,
    /// The world near the camera, for rays towards the sun.
    rays: RayWorld,
    ray_views: RayViews,
    clouds: CloudLayer,
    /// Position of each slot's chunk, as `Rgba32Sint` texels.
    origins: wgpu::Texture,
    origins_view: wgpu::TextureView,
    /// Which chunks are drawn in full, for far-away terrain to give way to
    /// them: a bit per chunk, see `chunk_drawn` in `world.wgsl`.
    drawn: wgpu::Texture,
    drawn_view: wgpu::TextureView,
    /// What `drawn` holds, and scratch space for this frame's.
    drawn_bits: [Vec<u32>; 2],
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
    /// Sampled by the clouds as well as drawn into.
    depth: wgpu::TextureView,
    size: (u32, u32),
    stats: RenderStats,
    culling: bool,
    /// The light the eye has got used to, and when.
    eye: Option<(Eye, std::time::Instant)>,
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

/// How the sun and the moon cast shadows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Shadows {
    /// None: the light of the sky says where the sun can't reach.
    #[default]
    Off,
    /// From a map of what the light sees, in cascades around the camera.
    Map,
    /// Near the camera, from a ray towards the light through the world for
    /// every point; farther, from the map.
    Rays,
}

/// How the world is lit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Lighting {
    /// As in classic block games without shaders: a fixed shade per side,
    /// a sky of picked colours, fog at the edge of the world.
    Classic,
    /// Sunlight and skylight through the air, haze, filmic colours.
    #[default]
    Atmospheric,
}

/// The clouds' textures and how they are drawn; see `clouds.rs`.
struct CloudLayer {
    obstacles: ObstacleMap,
    obstacle_texture: wgpu::Texture,
    obstacle_view: wgpu::TextureView,
    /// The cell the obstacle texture starts at.
    obstacle_origin: glam::IVec2,
    /// How deep in a patch of cloud each cell lies, for a seed.
    depths: Option<(u64, Vec<u8>)>,
    /// The seed and the cover the cells are for.
    cells_for: Option<(u64, f32)>,
    /// Which cells hold cloud; see `cloud_cells`.
    cells: wgpu::Texture,
    cells_view: wgpu::TextureView,
    /// Linear filtering, clamped and repeating.
    smooth_sampler: wgpu::Sampler,
    wrap_sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    /// Obstacles are taken into account at most so often.
    started: std::time::Instant,
}

/// Over how many frames each cascade is drawn: the nearest every frame, the
/// wider ones a part at a time.
const SHADOW_PARTS: [usize; CASCADES] = [1, 2, 4, 8];
/// Layers of the shadow map: the nearest cascade has one, the others two,
/// one shown while the other is being drawn.
const SHADOW_LAYERS: [[usize; 2]; CASCADES] = [[0, 0], [1, 2], [3, 4], [5, 6]];
const SHADOW_LAYER_COUNT: usize = 7;
/// The last cascade: far-away terrain casts its shadows.
const FAR_CASCADE: usize = CASCADES - 1;

/// Sun shadows: off by default, as they draw the world once more per cascade.
struct ShadowMap {
    enabled: bool,
    /// Whether rays towards the sun take the place of the near cascades.
    rays: bool,
    /// Sampled by the world pass, a layer per cascade.
    view: wgpu::TextureView,
    /// Drawn into, one per layer.
    layers: [wgpu::TextureView; SHADOW_LAYER_COUNT],
    /// Which of each cascade's two layers is shown.
    front: [usize; CASCADES],
    /// The cascades being drawn a part at a time.
    progress: [Option<Progress>; CASCADES],
    sampler: wgpu::Sampler,
    pipeline: wgpu::RenderPipeline,
    /// Draws far-away tiles.
    far_pipeline: wgpu::RenderPipeline,
    /// The globals and chunk positions, without the shadow map itself.
    globals_group: wgpu::BindGroup,
    cascades: [(wgpu::Buffer, wgpu::BindGroup); CASCADES],
    /// How each cascade was last drawn.
    drawn: [Option<Drawn>; CASCADES],
    /// Frames drawn so far.
    frames: u64,
    /// Room of chunk geometry that was replaced or taken away while a
    /// cascade being drawn a part at a time may still draw from it. It is
    /// given out again once no such cascade does.
    retired: Vec<Retired>,
}

/// A chunk's room in a page, and its slot, kept out of use for a while.
struct Retired {
    page: usize,
    space: Range<u32>,
    slot: u32,
    /// The first frame whose cascades don't draw from it.
    from: u64,
}

/// A cascade being drawn over a few frames.
struct Progress {
    drawn: Drawn,
    /// The chunks (or far-away tiles) as they were when it was started:
    /// those changed since keep their old geometry until it is done, see
    /// `ShadowMap::retired`.
    draws: Vec<Draw>,
    /// Whether `draws` are far-away tiles rather than chunks.
    far: bool,
    /// Parts drawn so far.
    part: usize,
    /// The frame it was started in.
    started: u64,
}

/// A cascade as drawn: the wider ones are drawn over a few frames and used
/// from where the camera was when they started.
#[derive(Clone, Copy, Debug)]
struct Drawn {
    view_proj: glam::Mat4,
    camera: DVec3,
    toward_light: Vec3,
    radius: f32,
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
        (width, height): (u32, u32),
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
                wgpu::BindGroupLayoutEntry {
                    binding: 7,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 8,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 10,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 11,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2Array,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 12,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 9,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 14,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 16,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 17,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D3,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 18,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Uint,
                        view_dimension: wgpu::TextureViewDimension::D2,
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
        let drawn = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("chunks drawn"),
            size: wgpu::Extent3d {
                width: DRAWN_WIDTH,
                height: DRAWN_WIDTH,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Uint,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let drawn_view = drawn.create_view(&Default::default());
        let sky_textures = upload_sky_textures(device, queue, [None, None]);
        let sky_table = upload_sky_table(device, queue);
        let shadows = ShadowMap::new(device, &shader, &globals, &origins_view);
        let mut rays = RayWorld::new();
        let ray_views = rays.set_enabled(device, false);
        let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("world"),
            bind_group_layouts: &[Some(&globals_layout)],
            immediate_size: 0,
        });
        let clouds = CloudLayer::new(device, &shader, &globals_layout, color_format);
        let globals_group = Self::globals_group(
            device,
            &globals_layout,
            &globals,
            [
                &block_textures,
                &origins_view,
                &sky_textures,
                &shadows.view,
                &clouds.cells_view,
                &sky_table,
                &clouds.obstacle_view,
                &drawn_view,
                &ray_views.solids,
                &ray_views.bricks,
                &ray_views.tops,
            ],
            [
                &sampler,
                &shadows.sampler,
                &clouds.smooth_sampler,
                &clouds.wrap_sampler,
            ],
        );
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
            // They give light: added to the sky, so a sun sinking into the
            // haze fades out instead of darkening.
            Some(wgpu::BlendState {
                color: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::SrcAlpha,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
                alpha: wgpu::BlendComponent {
                    src_factor: wgpu::BlendFactor::Zero,
                    dst_factor: wgpu::BlendFactor::One,
                    operation: wgpu::BlendOperation::Add,
                },
            }),
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
            sky_table,
            shadows,
            rays,
            ray_views,
            clouds,
            linear_output: color_format.is_srgb(),
            lighting: Lighting::default(),
            origins,
            origins_view,
            drawn,
            drawn_view,
            drawn_bits: [0, 1].map(|_| vec![0; (DRAWN_WIDTH * DRAWN_WIDTH) as usize]),
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
            size: (width, height),
            stats: RenderStats::default(),
            culling: true,
            eye: None,
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
            [
                &self.block_textures,
                &self.origins_view,
                &self.sky_textures,
                &self.shadows.view,
                &self.clouds.cells_view,
                &self.sky_table,
                &self.clouds.obstacle_view,
                &self.drawn_view,
                &self.ray_views.solids,
                &self.ray_views.bricks,
                &self.ray_views.tops,
            ],
            [
                &self.sampler,
                &self.shadows.sampler,
                &self.clouds.smooth_sampler,
                &self.clouds.wrap_sampler,
            ],
        );
    }

    /// How sun shadows are cast.
    pub(crate) fn set_shadows(&mut self, device: &wgpu::Device, shadows: Shadows) {
        let (map, rays) = (shadows != Shadows::Off, shadows == Shadows::Rays);
        if map == self.shadows.enabled && rays == self.shadows.rays {
            return;
        }
        if map != self.shadows.enabled {
            self.shadows.set_enabled(device, map);
        }
        if rays != self.shadows.rays {
            self.ray_views = self.rays.set_enabled(device, rays);
            self.shadows.rays = rays;
            self.shadows.drawn = [None; CASCADES];
            self.shadows.progress = Default::default();
        }
        self.rebuild_globals_group(device);
    }

    pub(crate) fn resize(&mut self, device: &wgpu::Device, width: u32, height: u32) {
        self.depth = depth_view(device, width, height);
        self.size = (width, height);
    }

    /// How the world is lit.
    pub(crate) fn set_lighting(&mut self, lighting: Lighting) {
        self.lighting = lighting;
    }

    pub(crate) fn upload(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        pos: ChunkPos,
        mesh: &ChunkMesh,
    ) {
        self.remove(queue, pos);
        self.rays.set(queue, pos, mesh.solids.clone());
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
            // The far cascade of the shadow map may still draw it.
            self.shadows.retired.push(Retired {
                page: gpu.page,
                space: gpu.space,
                slot: gpu.slot,
                from: self.shadows.frames + 1,
            });
        }
    }

    pub(crate) fn remove(&mut self, queue: &wgpu::Queue, pos: ChunkPos) {
        self.rays.set(queue, pos, None);
        if let Some(gpu) = self.chunks.remove(&pos).and_then(|entry| entry.gpu) {
            // A cascade of the shadow map started before may still draw it.
            self.shadows.retired.push(Retired {
                page: gpu.page,
                space: gpu.space,
                slot: gpu.slot,
                from: self.shadows.frames + 1,
            });
        }
    }

    /// Gives out again the room of chunk geometry that no cascade being
    /// drawn draws from any more.
    fn release_retired(&mut self) {
        let oldest = self
            .shadows
            .progress
            .iter()
            .flatten()
            .map(|progress| progress.started)
            .min();
        let (pages, slots) = (&mut self.pages, &mut self.slots);
        self.shadows.retired.retain(|retired| {
            let in_use = oldest.is_some_and(|started| started < retired.from);
            if !in_use {
                pages[retired.page].space.free(retired.space.clone());
                slots.give_back(retired.slot);
            }
            in_use
        });
    }

    pub(crate) fn clear(&mut self) {
        self.chunks.clear();
        self.lods.clear();
        self.pages.clear();
        self.slots = Slots::new(MAX_SLOTS);
        self.shadows.retired.clear();
        self.shadows.progress = Default::default();
        self.shadows.drawn = [None; CASCADES];
        self.rays.clear();
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

    /// Where blocks of `chunk` reach up towards the clouds; see
    /// `cloud_obstacles`.
    pub(crate) fn set_cloud_obstacles(&mut self, chunk: ChunkPos, tops: Option<ColumnTops>) {
        self.clouds.obstacles.set_chunk(chunk, tops);
    }

    pub(crate) fn set_far_cloud_obstacles(&mut self, tile: LodTilePos, tops: Option<Box<[i16]>>) {
        self.clouds.obstacles.set_tile(tile, tops);
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
        timer: &mut Option<GpuTimer>,
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
        if scene.lod_distance > 0.0 {
            self.mark_drawn(queue);
        }

        let selection = scene
            .target
            .map(|block| (block.0.as_dvec3() - camera).as_vec3());
        let camera_block = camera.floor();
        let camera_fract = (camera - camera_block).as_vec3();
        let camera_block = camera_block.as_ivec3();
        let cover = scene.clouds.map_or(0.0, |clouds| clouds.cover);
        let now = std::time::Instant::now();
        let seen = Eye::at(scene.eye_light);
        let eye = self.eye.map_or(seen, |(eye, at)| {
            eye.adapt(seen, now.duration_since(at).as_secs_f32())
        });
        self.eye = Some((eye, now));
        let sky = SkyLook::at(scene.time_of_day, eye, cover);
        if let Some(clouds) = &scene.clouds {
            self.clouds.update(queue, clouds, camera);
        }
        let selection = selection.unwrap_or(Vec3::ZERO);
        let mut globals = Std140::default();
        globals.matrix(view_proj);
        globals.matrix(view_proj.inverse());
        globals.floats([far * 0.7, far * 0.98, 0.0, HAZE]);
        globals.floats(selection.extend(0.0).to_array());
        let encode = if self.linear_output { 0.0 } else { 1.0 };
        globals.floats([width as f32, height as f32, sky.exposure, encode]);
        globals.ints(camera_block.extend(0).to_array());
        globals.floats(camera_fract.extend(0.0).to_array());
        globals.floats(sky.sun.extend(sky.day).to_array());
        // Shadows fade out as the sun or moon touches the horizon: long
        // shadows at sunset stay, as the light reddens and dims.
        let toward_light = sky.toward_light;
        let strength = smoothstep(0.0, 0.02, toward_light.y);
        let shadowed = self.shadows.enabled && strength > 0.0;
        let strength = if shadowed { strength } else { 0.0 };
        globals.floats(toward_light.extend(strength).to_array());
        globals.floats(sky.direct.extend(sky.ambient_floor).to_array());
        for term in sky.ambient {
            globals.floats(term.extend(0.0).to_array());
        }
        globals.floats([sky.sun_layer, sky.moon_layer, sky.moonlight, sky.stars]);
        let classic = self.lighting == Lighting::Classic;
        globals.floats([
            sky.cover,
            sky.eye,
            sky.discs,
            if classic { 1.0 } else { 0.0 },
        ]);
        globals.floats(sky.sun_disc.extend(0.0).to_array());
        let reach = reach(scene.view_distance, scene.lod_distance);
        self.shadows.frames += 1;
        if shadowed {
            self.draw_shadows(queue, encoder, camera, reach, toward_light, timer);
        } else {
            self.shadows.drawn = [None; CASCADES];
            self.shadows.progress = Default::default();
        }
        self.release_retired();
        for drawn in &self.shadows.drawn {
            // A cascade drawn a few frames ago is used from where the camera
            // was then.
            let matrix = drawn.map_or(glam::Mat4::IDENTITY, |drawn| {
                drawn.view_proj * glam::Mat4::from_translation((camera - drawn.camera).as_vec3())
            });
            globals.matrix(matrix);
        }
        globals.floats(reach);
        match &scene.clouds {
            Some(clouds) => {
                // Where the camera is over the air the wind has carried, in
                // the cell map, which repeats.
                let over = (glam::DVec2::new(camera.x, camera.z) - clouds.drift)
                    .rem_euclid(glam::DVec2::splat(CELL_MAP_PERIOD));
                let corner = self.clouds.obstacle_origin.as_dvec2() * f64::from(OBSTACLE_CELL)
                    - glam::DVec2::new(camera.x, camera.z);
                globals.floats([clouds.cover, clouds.density, CLOUD_BOTTOM, CLOUD_TOP]);
                globals.floats([over.x, over.y, corner.x, corner.y].map(|v| v as f32));
            }
            None => {
                for _ in 0..2 {
                    globals.floats([0.0; 4]);
                }
            }
        }
        globals.floats(sky.cloud_light.extend(0.0).to_array());
        globals.floats(sky.cloud_direct.extend(0.0).to_array());
        let shown = std::array::from_fn::<_, CASCADES, _>(|index| {
            SHADOW_LAYERS[index][self.shadows.front[index]] as f32
        });
        globals.floats(shown);
        let classic_sky = sky.classic;
        globals.floats(classic_sky.zenith.extend(classic_sky.stars).to_array());
        globals.floats(classic_sky.horizon.extend(0.0).to_array());
        globals.floats(classic_sky.glow.extend(0.0).to_array());
        globals.floats(classic_sky.fog.extend(0.0).to_array());
        globals.floats(classic_sky.light.extend(classic_sky.ambient).to_array());
        globals.floats(classic_sky.cloud.extend(0.0).to_array());
        let bounds = scene.bounds.unwrap_or_default();
        globals.ints(self.rays.update(queue, camera, bounds));
        debug_assert_eq!(globals.0.len() as u64, GLOBALS_SIZE);
        queue.write_buffer(&self.globals, 0, &globals.0);
        let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
            label: Some("world"),
            color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                view: target,
                depth_slice: None,
                resolve_target: None,
                ops: wgpu::Operations {
                    load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
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
            timestamp_writes: timer.as_mut().and_then(|timer| timer.pass("world")),
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

        if scene.clouds.is_some() {
            pass.set_pipeline(&self.clouds.pipeline);
            pass.draw(0..3, 0..1);
        }

        if scene.target.is_some() {
            pass.set_pipeline(&self.outline_pipeline);
            pass.draw(0..24, 0..1);
        }
        pass.set_pipeline(&self.crosshair_pipeline);
        pass.draw(0..12, 0..1);
    }

    /// Tells far-away terrain which chunks are drawn this frame: it gives
    /// way to them, and stands in for the others, still loading or out of
    /// sight of the walk through the chunks.
    fn mark_drawn(&mut self, queue: &wgpu::Queue) {
        let [shown, bits] = &mut self.drawn_bits;
        bits.fill(0);
        let wrap = DRAWN_WIDTH as i32 - 1;
        for pos in self
            .visible
            .iter()
            .filter(|pos| self.chunks.contains_key(pos))
        {
            let texel = (pos.0.z & wrap) * DRAWN_WIDTH as i32 + (pos.0.x & wrap);
            bits[texel as usize] |= 1 << (pos.0.y & 31);
        }
        if bits == shown {
            return;
        }
        std::mem::swap(shown, bits);
        let bytes: Vec<u8> = shown.iter().flat_map(|bits| bits.to_le_bytes()).collect();
        queue.write_texture(
            self.drawn.as_image_copy(),
            &bytes,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(DRAWN_WIDTH * 4),
                rows_per_image: None,
            },
            wgpu::Extent3d {
                width: DRAWN_WIDTH,
                height: DRAWN_WIDTH,
                depth_or_array_layers: 1,
            },
        );
    }

    /// Draws the cascades of the shadow map. The nearest is drawn whole
    /// every frame; the wider ones a part at a time, over 2 and 4 frames,
    /// into a second layer that is shown once complete, so no frame pays for
    /// a whole wide cascade. Cave culling doesn't apply: what the camera
    /// can't see may still cast a shadow into view.
    fn draw_shadows(
        &mut self,
        queue: &wgpu::Queue,
        encoder: &mut wgpu::CommandEncoder,
        camera: DVec3,
        reach: [f32; CASCADES],
        toward_light: Vec3,
        timer: &mut Option<GpuTimer>,
    ) {
        for (index, radius) in reach.into_iter().enumerate() {
            // Nothing farther than the cascade before to shade, or rays see
            // what the near cascades would.
            let unused = if index == FAR_CASCADE {
                radius <= reach[index - 1]
            } else {
                self.shadows.rays
            };
            if unused {
                self.shadows.drawn[index] = None;
                self.shadows.progress[index] = None;
                continue;
            }
            let parts = SHADOW_PARTS[index];
            // Drawn for another light, another reach or too far from here:
            // drawn whole, now.
            let stale = self.shadows.drawn[index].is_none_or(|drawn| {
                drawn.radius != radius
                    || drawn.toward_light.dot(toward_light) < 0.9999
                    || drawn.camera.distance(camera) > f64::from(radius) * 0.1
            });
            let whole = stale || parts == 1;
            if whole || self.shadows.progress[index].is_none() {
                let view_proj = cascade(camera, radius, toward_light, margin(index));
                // The last cascade draws far-away terrain, which is there
                // near the camera too, unless there is none.
                let far = index == FAR_CASCADE && !self.lods.is_empty();
                let draws = if far {
                    self.far_casters(view_proj, camera)
                } else {
                    self.casters(view_proj, camera)
                };
                self.shadows.progress[index] = Some(Progress {
                    drawn: Drawn {
                        view_proj,
                        camera,
                        toward_light,
                        radius,
                    },
                    draws,
                    far,
                    part: 0,
                    started: self.shadows.frames,
                });
            }
            let progress = self.shadows.progress[index].as_mut().expect("just started");
            // Chunks are placed relative to where the camera is now; the
            // cascade was laid out from where it was when it was started.
            let matrix = progress.drawn.view_proj
                * glam::Mat4::from_translation((camera - progress.drawn.camera).as_vec3());
            let mut uniform = Std140::default();
            uniform.matrix(matrix);
            queue.write_buffer(&self.shadows.cascades[index].0, 0, &uniform.0);
            let count = progress.draws.len();
            let (first, part) = (progress.part, if whole { parts } else { 1 });
            let range = (count * first / parts)..(count * (first + part) / parts);
            let back = SHADOW_LAYERS[index][1 - self.shadows.front[index]];
            self.stats.shadow_draw_calls += range.len();

            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("shadow map"),
                color_attachments: &[],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &self.shadows.layers[back],
                    depth_ops: Some(wgpu::Operations {
                        load: if first == 0 {
                            wgpu::LoadOp::Clear(1.0)
                        } else {
                            wgpu::LoadOp::Load
                        },
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: timer.as_mut().and_then(|timer| timer.pass("shadows")),
                ..Default::default()
            });
            pass.set_pipeline(if progress.far {
                &self.shadows.far_pipeline
            } else {
                &self.shadows.pipeline
            });
            pass.set_bind_group(0, &self.shadows.globals_group, &[]);
            pass.set_bind_group(1, &self.shadows.cascades[index].1, &[]);
            let mut bound = None;
            for draw in &progress.draws[range] {
                if bound != Some(draw.page) {
                    pass.set_vertex_buffer(0, self.pages[draw.page].buffer.slice(..));
                    bound = Some(draw.page);
                }
                pass.draw(0..4, draw.quads.clone());
            }
            drop(pass);

            progress.part = first + part;
            if progress.part >= parts {
                // Complete: show it.
                self.shadows.drawn[index] = Some(progress.drawn);
                self.shadows.front[index] = 1 - self.shadows.front[index];
                self.shadows.progress[index] = None;
            }
        }
    }

    /// Every far-away tile the light sees in a cascade, as draws in buffer
    /// order.
    fn far_casters(&self, view_proj: glam::Mat4, camera: DVec3) -> Vec<Draw> {
        let frustum = Frustum::new(view_proj);
        let mut casters: Vec<&GpuLod> = self
            .lods
            .iter()
            .filter(|(pos, gpu)| {
                let (x, z) = pos.origin();
                let min = DVec3::new(f64::from(x), f64::from(gpu.heights.start - 1), f64::from(z));
                let size = DVec3::new(
                    f64::from(LOD_TILE_SIZE),
                    f64::from(gpu.heights.end - gpu.heights.start + 2),
                    f64::from(LOD_TILE_SIZE),
                );
                let min = (min - camera).as_vec3();
                frustum.intersects_box(min, min + size.as_vec3())
            })
            .map(|(_, gpu)| gpu)
            .collect();
        casters.sort_by_key(|gpu| (gpu.page, gpu.space.start));
        let mut draws = Vec::new();
        for gpu in casters {
            push_draw(&mut draws, gpu.page, gpu.space.clone());
        }
        draws
    }

    /// Every chunk the light sees in a cascade, as draws in buffer order, so
    /// neighbours merge into one.
    fn casters(&self, view_proj: glam::Mat4, camera: DVec3) -> Vec<Draw> {
        let frustum = Frustum::new(view_proj);
        let mut casters: Vec<&GpuChunk> = self
            .chunks
            .iter()
            .filter_map(|(pos, entry)| {
                let min = (pos.origin().0.as_dvec3() - camera).as_vec3();
                let seen = frustum.intersects_box(min, min + Vec3::splat(CHUNK_SIZE as f32));
                entry.gpu.as_ref().filter(|_| seen)
            })
            .collect();
        casters.sort_by_key(|gpu| (gpu.page, gpu.quads.start));
        let mut draws = Vec::new();
        for gpu in casters {
            push_draw(&mut draws, gpu.page, gpu.quads.clone());
        }
        draws
    }

    fn globals_group(
        device: &wgpu::Device,
        layout: &wgpu::BindGroupLayout,
        globals: &wgpu::Buffer,
        [
            textures,
            origins,
            sky,
            shadows,
            cells,
            sky_table,
            obstacles,
            drawn,
            ray_solids,
            ray_bricks,
            ray_tops,
        ]: [&wgpu::TextureView; 11],
        [sampler, shadow_sampler, smooth_sampler, wrap_sampler]: [&wgpu::Sampler; 4],
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
                wgpu::BindGroupEntry {
                    binding: 7,
                    resource: wgpu::BindingResource::TextureView(cells),
                },
                wgpu::BindGroupEntry {
                    binding: 8,
                    resource: wgpu::BindingResource::Sampler(smooth_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 10,
                    resource: wgpu::BindingResource::Sampler(wrap_sampler),
                },
                wgpu::BindGroupEntry {
                    binding: 11,
                    resource: wgpu::BindingResource::TextureView(shadows),
                },
                wgpu::BindGroupEntry {
                    binding: 9,
                    resource: wgpu::BindingResource::TextureView(sky_table),
                },
                wgpu::BindGroupEntry {
                    binding: 12,
                    resource: wgpu::BindingResource::TextureView(obstacles),
                },
                wgpu::BindGroupEntry {
                    binding: 14,
                    resource: wgpu::BindingResource::TextureView(drawn),
                },
                wgpu::BindGroupEntry {
                    binding: 16,
                    resource: wgpu::BindingResource::TextureView(ray_solids),
                },
                wgpu::BindGroupEntry {
                    binding: 17,
                    resource: wgpu::BindingResource::TextureView(ray_bricks),
                },
                wgpu::BindGroupEntry {
                    binding: 18,
                    resource: wgpu::BindingResource::TextureView(ray_tops),
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

impl CloudLayer {
    fn new(
        device: &wgpu::Device,
        shader: &wgpu::ShaderModule,
        globals_layout: &wgpu::BindGroupLayout,
        color_format: wgpu::TextureFormat,
    ) -> Self {
        let texture = |label, size: wgpu::Extent3d, dimension| {
            let texture = device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size,
                mip_level_count: 1,
                sample_count: 1,
                dimension,
                format: wgpu::TextureFormat::R8Unorm,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            });
            let view = texture.create_view(&Default::default());
            (texture, view)
        };
        let square = |size| wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: 1,
        };
        // With smaller levels, each the one before halved: blurrier cloud
        // shadows far below their clouds.
        let cells = device.create_texture(&wgpu::TextureDescriptor {
            label: Some("cloud cells"),
            size: square(CELL_MAP_SIZE),
            mip_level_count: CELL_MAP_SIZE.ilog2() + 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R8Unorm,
            usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
            view_formats: &[],
        });
        let cells_view = cells.create_view(&Default::default());
        let (obstacle_texture, obstacle_view) = texture(
            "cloud obstacles",
            square(OBSTACLE_SIZE),
            wgpu::TextureDimension::D2,
        );
        let sampler = |label, address_mode| {
            device.create_sampler(&wgpu::SamplerDescriptor {
                label: Some(label),
                address_mode_u: address_mode,
                address_mode_v: address_mode,
                address_mode_w: address_mode,
                mag_filter: wgpu::FilterMode::Linear,
                min_filter: wgpu::FilterMode::Linear,
                mipmap_filter: wgpu::MipmapFilterMode::Linear,
                ..Default::default()
            })
        };

        let pipeline = |label,
                        layouts: &[Option<&wgpu::BindGroupLayout>],
                        (vertex, fragment),
                        topology,
                        targets: &[Option<wgpu::ColorTargetState>],
                        depth_stencil| {
            let layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(label),
                bind_group_layouts: layouts,
                immediate_size: 0,
            });
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(vertex),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive: wgpu::PrimitiveState {
                    topology,
                    ..Default::default()
                },
                depth_stencil,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: shader,
                    entry_point: Some(fragment),
                    compilation_options: Default::default(),
                    targets,
                }),
                multiview_mask: None,
                cache: None,
            })
        };
        // Tested against the world, never written.
        let depth_test = Some(wgpu::DepthStencilState {
            format: DEPTH_FORMAT,
            depth_write_enabled: Some(false),
            depth_compare: Some(wgpu::CompareFunction::LessEqual),
            stencil: Default::default(),
            bias: Default::default(),
        });
        let cloud_pipeline = pipeline(
            "clouds",
            &[Some(globals_layout)],
            ("sky_vertex", "cloud_fragment"),
            wgpu::PrimitiveTopology::TriangleList,
            &[Some(wgpu::ColorTargetState {
                format: color_format,
                blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                write_mask: wgpu::ColorWrites::ALL,
            })],
            depth_test.clone(),
        );

        let smooth_sampler = sampler("clouds, clamped", wgpu::AddressMode::ClampToEdge);
        let wrap_sampler = sampler("clouds, repeating", wgpu::AddressMode::Repeat);
        Self {
            obstacles: ObstacleMap::default(),
            obstacle_texture,
            obstacle_view,
            obstacle_origin: glam::IVec2::ZERO,
            depths: None,
            cells_for: None,
            cells,
            cells_view,
            smooth_sampler,
            wrap_sampler,
            pipeline: cloud_pipeline,
            started: std::time::Instant::now(),
        }
    }

    /// Brings the textures up to date: the cells for the sky's seed and
    /// cover, the obstacles around the camera.
    fn update(&mut self, queue: &wgpu::Queue, sky: &CloudSky, camera: DVec3) {
        if self.cells_for != Some((sky.seed, sky.cover)) {
            let depths = match self.depths.take() {
                Some((seed, depths)) if seed == sky.seed => depths,
                _ => cloud_depths(sky.seed),
            };
            let mut level = cloud_cells(&depths, sky.cover);
            let mut size = CELL_MAP_SIZE;
            for mip in 0.. {
                queue.write_texture(
                    wgpu::TexelCopyTextureInfo {
                        texture: &self.cells,
                        mip_level: mip,
                        origin: wgpu::Origin3d::ZERO,
                        aspect: wgpu::TextureAspect::All,
                    },
                    &level,
                    wgpu::TexelCopyBufferLayout {
                        offset: 0,
                        bytes_per_row: Some(size),
                        rows_per_image: Some(size),
                    },
                    wgpu::Extent3d {
                        width: size,
                        height: size,
                        depth_or_array_layers: 1,
                    },
                );
                if size == 1 {
                    break;
                }
                level = halve(&level, size as usize);
                size /= 2;
            }
            self.depths = Some((sky.seed, depths));
            self.cells_for = Some((sky.seed, sky.cover));
        }
        let time = self.started.elapsed().as_secs_f64();
        if let Some(image) = self.obstacles.update(camera, time) {
            write_r8(
                queue,
                &self.obstacle_texture,
                &image.texels,
                (OBSTACLE_SIZE, OBSTACLE_SIZE, 1),
            );
            self.obstacle_origin = image.origin;
        }
    }
}

/// The average of each 2×2 texels of a square single-channel image `size`
/// texels wide.
fn halve(image: &[u8], size: usize) -> Vec<u8> {
    let half = size / 2;
    (0..half * half)
        .map(|i| {
            let (x, y) = (i % half * 2, i / half * 2);
            let at = |dx: usize, dy: usize| u32::from(image[(y + dy) * size + x + dx]);
            ((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8
        })
        .collect()
}

/// Fills a single-channel texture of `width`, `height` and `depth` texels.
fn write_r8(
    queue: &wgpu::Queue,
    texture: &wgpu::Texture,
    texels: &[u8],
    (width, height, depth): (u32, u32, u32),
) {
    queue.write_texture(
        texture.as_image_copy(),
        texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(width),
            rows_per_image: Some(height),
        },
        wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: depth,
        },
    );
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
        let cascades = std::array::from_fn(|_| {
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
        let pipeline = |label, entry_point| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&layout),
                vertex: wgpu::VertexState {
                    module: shader,
                    entry_point: Some(entry_point),
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
            })
        };
        let far_pipeline = pipeline("shadow map, far terrain", "lod_shadow_vertex");
        let pipeline = pipeline("shadow map", "shadow_vertex");
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
            rays: false,
            view,
            layers,
            sampler,
            pipeline,
            far_pipeline,
            globals_group,
            cascades,
            drawn: [None; CASCADES],
            front: [0; CASCADES],
            progress: Default::default(),
            frames: 0,
            retired: Vec::new(),
        }
    }

    /// Off, the map shrinks to a texel per cascade.
    fn set_enabled(&mut self, device: &wgpu::Device, enabled: bool) {
        let size = if enabled { SHADOW_MAP_SIZE } else { 1 };
        (self.view, self.layers) = shadow_texture(device, size);
        self.enabled = enabled;
        self.drawn = [None; CASCADES];
        self.progress = Default::default();
    }
}

/// A depth texture with a layer per cascade, as one view to sample and one
/// per layer to draw into.
fn shadow_texture(
    device: &wgpu::Device,
    size: u32,
) -> (wgpu::TextureView, [wgpu::TextureView; SHADOW_LAYER_COUNT]) {
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("shadow map"),
        size: wgpu::Extent3d {
            width: size,
            height: size,
            depth_or_array_layers: SHADOW_LAYER_COUNT as u32,
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
    let layers = std::array::from_fn(|layer| {
        texture.create_view(&wgpu::TextureViewDescriptor {
            dimension: Some(wgpu::TextureViewDimension::D2),
            base_array_layer: layer as u32,
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

/// The sky table (see `atmosphere.rs`) as a 3-D texture: directions around
/// from the light, heights above the horizon, heights of the light.
fn upload_sky_table(device: &wgpu::Device, queue: &wgpu::Queue) -> wgpu::TextureView {
    let size = wgpu::Extent3d {
        width: SKY_AZIMUTHS,
        height: SKY_ELEVATIONS,
        depth_or_array_layers: SKY_LIGHTS,
    };
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some("sky table"),
        size,
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D3,
        format: wgpu::TextureFormat::Rgba16Float,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    let texels: Vec<u8> = SkyTables::get()
        .sky
        .iter()
        .flat_map(|half| half.to_le_bytes())
        .collect();
    queue.write_texture(
        texture.as_image_copy(),
        &texels,
        wgpu::TexelCopyBufferLayout {
            offset: 0,
            bytes_per_row: Some(SKY_AZIMUTHS * 8),
            rows_per_image: Some(SKY_ELEVATIONS),
        },
        size,
    );
    texture.create_view(&Default::default())
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

    /// The OpenGL backend turns the shader into GLSL, which can't do
    /// everything WGSL can: every stage must translate.
    #[test]
    fn shader_translates_for_opengl() {
        let module = naga::front::wgsl::parse_str(include_str!("world.wgsl")).unwrap();
        let info = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::empty(),
        )
        .validate(&module)
        .unwrap();
        let options = naga::back::glsl::Options {
            version: naga::back::glsl::Version::Desktop(330),
            ..Default::default()
        };
        for entry in &module.entry_points {
            let pipeline = naga::back::glsl::PipelineOptions {
                shader_stage: entry.stage,
                entry_point: entry.name.clone(),
                multiview: None,
            };
            let mut glsl = String::new();
            naga::back::glsl::Writer::new(
                &mut glsl,
                &module,
                &info,
                &options,
                &pipeline,
                naga::proc::BoundsCheckPolicies::default(),
            )
            .and_then(|mut writer| writer.write())
            .unwrap_or_else(|error| panic!("{}: {error}", entry.name));
        }
    }
}
