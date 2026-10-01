//! Renderer on top of wgpu.
//!
//! wgpu types never leave this crate: the rest of the engine sees the
//! [`Renderer`], the [`Camera`] and the meshing helpers.

mod arena;
mod camera;
mod clouds;
mod culling;
mod lod;
mod mesh;
mod mesher;
mod shadows;
mod sky;
mod textures;
mod visibility;
mod world_pass;

use std::fmt;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ruda_core::{BlockPos, ChunkPos, Content, Light, WorldBounds};
use tracing::{info, warn};
use wgpu::rwh::{HasDisplayHandle, HasWindowHandle};

pub use camera::{Camera, Frustum};
pub use clouds::{
    CLOUD_BOTTOM, CLOUD_CELL, CLOUD_THICKNESS, CloudSky, WIND, cloud_at, cloud_obstacles,
    far_cloud_obstacles,
};
pub use lod::{LodMesh, LodQuad, mesh_lod};
pub use mesh::{BlockFaces, ChunkMesh, ModelVertex, PaddedChunk, Quad, mesh_chunk};
pub use mesher::ChunkMesher;
pub use textures::BlockTextures;
pub use visibility::Visibility;

use world_pass::WorldPass;

/// What fills the screen behind the interface.
#[derive(Clone, Copy, Debug)]
pub enum Backdrop<'a> {
    World(&'a Scene),
    /// A plain sRGB colour, for menus outside a game.
    Color([u8; 3]),
}

/// An egui frame to draw over the backdrop.
#[derive(Clone, Copy)]
pub struct UiFrame<'a> {
    pub primitives: &'a [egui::ClippedPrimitive],
    pub textures: &'a egui::TexturesDelta,
    pub pixels_per_point: f32,
}

impl fmt::Debug for UiFrame<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("UiFrame")
            .field("primitives", &self.primitives.len())
            .field("pixels_per_point", &self.pixels_per_point)
            .finish_non_exhaustive()
    }
}

/// A view of the world to draw.
#[derive(Clone, Copy, Debug)]
pub struct Scene {
    pub camera: Camera,
    /// The block under the crosshair, outlined.
    pub target: Option<BlockPos>,
    /// How far the world is drawn, in blocks; fog hides the edge.
    pub view_distance: f32,
    /// The heights the world spans; above it there is only sky.
    pub bounds: Option<WorldBounds>,
    /// Fraction of the day gone: 0 sunrise, 0.25 noon, 0.5 sunset,
    /// 0.75 midnight.
    pub time_of_day: f32,
    /// The light where the camera is; from caves the sky looks dark.
    pub eye_light: Light,
    /// How far the far-away look of the world reaches, in blocks; 0 for
    /// none.
    pub lod_distance: f32,
    /// `None` for a sky without clouds.
    pub clouds: Option<CloudSky>,
}

/// What the last frame drew.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct RenderStats {
    /// Chunks with geometry on the GPU.
    pub chunks: usize,
    /// Chunks drawn after culling.
    pub drawn_chunks: usize,
    pub draw_calls: usize,
    pub quads: u64,
    /// Draw calls into the shadow map.
    pub shadow_draw_calls: usize,
}

impl fmt::Display for RenderStats {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} of {} chunks drawn in {} draw calls, {} quads",
            self.drawn_chunks, self.chunks, self.draw_calls, self.quads
        )?;
        if self.shadow_draw_calls > 0 {
            write!(f, "; {} draw calls for shadows", self.shadow_draw_calls)?;
        }
        Ok(())
    }
}

/// Graphics API to render with.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum GpuBackend {
    /// Vulkan, Metal or DX12, falling back to OpenGL when none of them works.
    #[default]
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl GpuBackend {
    /// Backend sets to try, in order.
    fn attempts(self) -> &'static [wgpu::Backends] {
        match self {
            Self::Auto => &[wgpu::Backends::PRIMARY, wgpu::Backends::GL],
            Self::Vulkan => &[wgpu::Backends::VULKAN],
            Self::Metal => &[wgpu::Backends::METAL],
            Self::Dx12 => &[wgpu::Backends::DX12],
            Self::Gl => &[wgpu::Backends::GL],
        }
    }
}

/// Owns the GPU device and the window surface.
pub struct Renderer {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    /// `None` for a renderer that only draws off-screen.
    window: Option<Arc<dyn wgpu::WindowHandle>>,
    config: wgpu::SurfaceConfiguration,
    /// `None` while suspended: mobile platforms destroy the native window.
    surface: Option<wgpu::Surface<'static>>,
    world: WorldPass,
    /// How long the last frame waited for the window to hand out an image.
    surface_wait: std::time::Duration,
    ui: egui_wgpu::Renderer,
    /// egui blends in gamma space, so where the GPU allows it the interface
    /// draws through a non-sRGB view of the frame.
    ui_format: wgpu::TextureFormat,
}

impl Renderer {
    /// Picks an adapter for `backend` and prepares the window surface.
    ///
    /// `display` is the platform display connection (winit's `OwnedDisplayHandle`);
    /// the OpenGL backend needs it to present, especially on Wayland.
    pub async fn new<D, W>(
        display: D,
        window: Arc<W>,
        width: u32,
        height: u32,
        backend: GpuBackend,
    ) -> Result<Self>
    where
        D: HasDisplayHandle + fmt::Debug + Clone + Send + Sync + 'static,
        W: HasWindowHandle + Send + Sync + 'static,
    {
        let window: Arc<dyn wgpu::WindowHandle> = window;
        let mut last_error = None;
        for &backends in backend.attempts() {
            match Self::with_backends(display.clone(), window.clone(), backends, width, height)
                .await
            {
                Ok(renderer) => return Ok(renderer),
                Err(error) => {
                    warn!(?backends, "graphics backend unavailable: {error:#}");
                    last_error = Some(error);
                }
            }
        }
        let error = last_error.unwrap_or_else(|| anyhow::anyhow!("no backends to try"));
        Err(error.context(format!("failed to initialize graphics ({backend:?})")))
    }

    async fn with_backends<D>(
        display: D,
        window: Arc<dyn wgpu::WindowHandle>,
        backends: wgpu::Backends,
        width: u32,
        height: u32,
    ) -> Result<Self>
    where
        D: HasDisplayHandle + fmt::Debug + Send + Sync + 'static,
    {
        let mut desc = wgpu::InstanceDescriptor::new_with_display_handle(Box::new(display));
        desc.backends = backends;
        let instance = wgpu::Instance::new(desc);
        let surface = create_surface(&instance, &window)?;

        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .context("no compatible GPU adapter")?;

        // Gameplay must run on WebGL2-class hardware, so ask for exactly those
        // limits and let wgpu reject anything that goes beyond them.
        let required_limits =
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        let device = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ruda"),
                required_limits,
                ..Default::default()
            })
            .await
            .context("failed to create the GPU device")?;

        let mut config = surface
            .get_default_config(&adapter, width, height)
            .context("the adapter cannot present to this window")?;
        let caps = surface.get_capabilities(&adapter);
        if let Some(format) = caps.formats.iter().copied().find(|f| f.is_srgb()) {
            config.format = format;
        }
        config.present_mode = wgpu::PresentMode::AutoVsync;
        let renderer = Self::with_device(
            instance,
            adapter,
            device,
            config,
            Some(window),
            Some(surface),
        );
        renderer.configure_surface();
        Ok(renderer)
    }

    /// A renderer without a window, for drawing into images with
    /// [`Renderer::capture`]: tests and tools.
    pub async fn headless(width: u32, height: u32) -> Result<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions::default())
            .await
            .context("no GPU adapter")?;
        let required_limits =
            wgpu::Limits::downlevel_webgl2_defaults().using_resolution(adapter.limits());
        let device = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("ruda"),
                required_limits,
                ..Default::default()
            })
            .await
            .context("failed to create the GPU device")?;
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            width,
            height,
            present_mode: wgpu::PresentMode::AutoVsync,
            desired_maximum_frame_latency: 2,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: Vec::new(),
            color_space: Default::default(),
        };
        Ok(Self::with_device(
            instance, adapter, device, config, None, None,
        ))
    }

    fn with_device(
        instance: wgpu::Instance,
        adapter: wgpu::Adapter,
        (device, queue): (wgpu::Device, wgpu::Queue),
        mut config: wgpu::SurfaceConfiguration,
        window: Option<Arc<dyn wgpu::WindowHandle>>,
        surface: Option<wgpu::Surface<'static>>,
    ) -> Self {
        let (width, height) = (config.width, config.height);
        let reinterpret =
            wgpu::DownlevelFlags::SURFACE_VIEW_FORMATS | wgpu::DownlevelFlags::VIEW_FORMATS;
        let ui_format = if adapter
            .get_downlevel_capabilities()
            .flags
            .contains(reinterpret)
        {
            config.format.remove_srgb_suffix()
        } else {
            config.format
        };
        if ui_format != config.format {
            config.view_formats.push(ui_format);
        }

        let info = adapter.get_info();
        info!(
            backend = %info.backend,
            adapter = %info.name,
            device_type = ?info.device_type,
            driver = %info.driver,
            driver_info = %info.driver_info,
            format = ?config.format,
            "GPU ready"
        );

        let world = WorldPass::new(&device, &queue, config.format, width, height);
        let ui = egui_wgpu::Renderer::new(
            &device,
            ui_format,
            egui_wgpu::RendererOptions {
                msaa_samples: 1,
                ..Default::default()
            },
        );
        Self {
            instance,
            adapter,
            device,
            queue,
            window,
            config,
            surface,
            world,
            surface_wait: std::time::Duration::ZERO,
            ui,
            ui_format,
        }
    }

    /// Short description of the active GPU, e.g. "Metal · Apple M1".
    pub fn adapter_summary(&self) -> String {
        let info = self.adapter.get_info();
        format!("{} · {}", info.backend, info.name)
    }

    /// Call when the window's physical size changes.
    pub fn resize(&mut self, width: u32, height: u32) {
        let max = self.device.limits().max_texture_dimension_2d;
        self.config.width = width.min(max);
        self.config.height = height.min(max);
        self.configure_surface();
        if self.config.width > 0 && self.config.height > 0 {
            self.world
                .resize(&self.device, self.config.width, self.config.height);
        }
    }

    /// Uploads the block textures of `content` and returns, for meshing, the
    /// texture layer of every block face.
    pub fn load_block_textures(&mut self, content: &Content) -> BlockFaces {
        let max_layers = self.device.limits().max_texture_array_layers;
        let textures = BlockTextures::load(content, max_layers);
        self.world
            .set_textures(&self.device, &self.queue, &textures);
        BlockFaces::new(content.blocks(), |id| textures.layer(id))
    }

    /// The sun and moon images, PNG-encoded, up to 32 pixels square.
    pub fn set_sky_textures(&mut self, sun: &[u8], moon: &[u8]) {
        self.world
            .set_sky_textures(&self.device, &self.queue, sun, moon);
    }

    /// Replaces the geometry drawn for a chunk.
    pub fn upload_chunk(&mut self, pos: ChunkPos, mesh: &ChunkMesh) {
        self.world.upload(&self.device, &self.queue, pos, mesh);
    }

    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        self.world.remove(pos);
    }

    /// Replaces the geometry drawn for a tile of far-away terrain.
    pub fn upload_lod(&mut self, pos: ruda_world::lod::LodTilePos, mesh: &LodMesh) {
        self.world.upload_lod(&self.device, &self.queue, pos, mesh);
    }

    pub fn remove_lod(&mut self, pos: ruda_world::lod::LodTilePos) {
        self.world.remove_lod(pos);
    }

    /// Where the blocks of a chunk reach up into the clouds, which part
    /// around them; see [`cloud_obstacles`].
    pub fn set_cloud_obstacles(&mut self, chunk: ChunkPos, columns: u64) {
        self.world.set_cloud_obstacles(chunk, columns);
    }

    /// The same for far-away terrain, see [`far_cloud_obstacles`]; `None`
    /// once the tile is gone.
    pub fn set_far_cloud_obstacles(
        &mut self,
        tile: ruda_world::lod::LodTilePos,
        rows: Option<[u64; ruda_world::lod::LOD_TILE_CELLS]>,
    ) {
        self.world.set_far_cloud_obstacles(tile, rows);
    }

    /// Forgets every chunk, for leaving a world.
    pub fn clear_chunks(&mut self) {
        self.world.clear();
    }

    /// The largest texture the interface may upload, in pixels per side.
    pub fn max_texture_side(&self) -> usize {
        self.device.limits().max_texture_dimension_2d as usize
    }

    /// Chunks with geometry on the GPU.
    pub fn chunk_count(&self) -> usize {
        self.world.chunk_count()
    }

    /// How long the last frame waited for the window to hand out an image to
    /// draw into: with vertical sync, or when frames come faster than the
    /// system composites them.
    pub fn surface_wait(&self) -> std::time::Duration {
        self.surface_wait
    }

    /// Turns sun shadows on or off. They draw the world once more for each
    /// of the shadow map's two cascades, so they are for stronger GPUs.
    pub fn set_shadows(&mut self, enabled: bool) {
        self.world.set_shadows(&self.device, enabled);
    }

    /// Turns skipping chunks hidden behind solid ground and faces turned away
    /// from the camera on or off. It is on by default; turning it off shows
    /// whether it hides anything that should be seen.
    pub fn set_culling(&mut self, culling: bool) {
        self.world.set_culling(culling);
    }

    /// What the last frame of the world drew.
    pub fn stats(&self) -> RenderStats {
        self.world.stats()
    }

    /// Draws a frame into an off-screen image instead of the window and
    /// returns its width, height and RGBA pixels.
    pub fn capture(
        &mut self,
        backdrop: Backdrop<'_>,
        ui: Option<&UiFrame<'_>>,
    ) -> Result<(u32, u32, Vec<u8>)> {
        let (width, height) = (self.config.width.max(1), self.config.height.max(1));
        let format = self.config.format;
        let size = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("screenshot"),
            size,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[self.ui_format],
        });
        let row = (width * 4).next_multiple_of(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT);
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("screenshot"),
            size: u64::from(row * height),
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("screenshot"),
            });
        self.update_ui_textures(ui);
        let ui_commands = self.encode(&mut encoder, &texture, (width, height), backdrop, ui);
        encoder.copy_texture_to_buffer(
            texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(row),
                    rows_per_image: Some(height),
                },
            },
            size,
        );
        self.queue
            .submit(ui_commands.into_iter().chain([encoder.finish()]));
        self.free_ui_textures(ui);

        buffer.map_async(wgpu::MapMode::Read, .., |_| {});
        self.device.poll(wgpu::PollType::Wait {
            submission_index: None,
            timeout: None,
        })?;
        let mapped = buffer.get_mapped_range(..)?;
        let mut pixels = Vec::with_capacity((width * height * 4) as usize);
        for line in mapped.chunks(row as usize) {
            pixels.extend_from_slice(&line[..(width * 4) as usize]);
        }
        if matches!(
            format,
            wgpu::TextureFormat::Bgra8Unorm | wgpu::TextureFormat::Bgra8UnormSrgb
        ) {
            for pixel in pixels.as_chunks_mut::<4>().0 {
                pixel.swap(0, 2);
            }
        }
        Ok((width, height, pixels))
    }

    /// Releases the surface; call when the platform suspends the app.
    pub fn suspend(&mut self) {
        self.surface = None;
    }

    /// Recreates the surface released by [`Renderer::suspend`].
    pub fn resume(&mut self) -> Result<()> {
        if self.surface.is_none()
            && let Some(window) = &self.window
        {
            self.surface = Some(create_surface(&self.instance, window)?);
            self.configure_surface();
        }
        Ok(())
    }

    /// Draws a frame and returns whether it reached the screen: nothing is
    /// presented while the window is hidden, zero-sized or being reconfigured.
    /// `pre_present` runs right before presenting (winit wants
    /// `Window::pre_present_notify` there).
    pub fn render(
        &mut self,
        backdrop: Backdrop<'_>,
        ui: Option<&UiFrame<'_>>,
        pre_present: impl FnOnce(),
    ) -> Result<bool> {
        // egui sends each texture change only once, so keep them even when
        // the frame is skipped.
        self.update_ui_textures(ui);
        let presented = self.present(backdrop, ui, pre_present);
        self.free_ui_textures(ui);
        presented
    }

    fn present(
        &mut self,
        backdrop: Backdrop<'_>,
        ui: Option<&UiFrame<'_>>,
        pre_present: impl FnOnce(),
    ) -> Result<bool> {
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(false);
        }
        let Some(surface) = &self.surface else {
            return Ok(false);
        };

        let acquire = std::time::Instant::now();
        let current = surface.get_current_texture();
        self.surface_wait = acquire.elapsed();
        let (frame, suboptimal) = match current {
            wgpu::CurrentSurfaceTexture::Success(frame) => (frame, false),
            wgpu::CurrentSurfaceTexture::Suboptimal(frame) => (frame, true),
            wgpu::CurrentSurfaceTexture::Timeout | wgpu::CurrentSurfaceTexture::Occluded => {
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Outdated => {
                self.configure_surface();
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Lost => {
                warn!("window surface lost, recreating it");
                if let Some(window) = &self.window {
                    self.surface = Some(create_surface(&self.instance, window)?);
                    self.configure_surface();
                }
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                bail!("validation error while acquiring a frame")
            }
        };

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        let size = (self.config.width, self.config.height);
        let ui_commands = self.encode(&mut encoder, &frame.texture, size, backdrop, ui);
        self.queue
            .submit(ui_commands.into_iter().chain([encoder.finish()]));

        pre_present();
        self.queue.present(frame);
        if suboptimal {
            self.configure_surface();
        }
        Ok(true)
    }

    /// Turns waiting for the display's refresh on or off.
    pub fn set_vsync(&mut self, vsync: bool) {
        self.config.present_mode = if vsync {
            wgpu::PresentMode::AutoVsync
        } else {
            wgpu::PresentMode::AutoNoVsync
        };
        self.configure_surface();
    }

    /// Records the backdrop and the interface into `encoder`. Returns extra
    /// command buffers egui needs submitted before it.
    fn encode(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        target: &wgpu::Texture,
        (width, height): (u32, u32),
        backdrop: Backdrop<'_>,
        ui: Option<&UiFrame<'_>>,
    ) -> Vec<wgpu::CommandBuffer> {
        let format = self.config.format;
        let view = &target.create_view(&wgpu::TextureViewDescriptor::default());
        match backdrop {
            Backdrop::World(scene) => self.world.draw(
                &self.device,
                &self.queue,
                encoder,
                view,
                scene,
                (width, height),
            ),
            Backdrop::Color(rgb) => {
                encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                    label: Some("backdrop"),
                    color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                        view,
                        depth_slice: None,
                        resolve_target: None,
                        ops: wgpu::Operations {
                            load: wgpu::LoadOp::Clear(color(
                                rgb.map(|c| f64::from(c) / 255.0),
                                format,
                            )),
                            store: wgpu::StoreOp::Store,
                        },
                    })],
                    ..Default::default()
                });
            }
        }

        let Some(ui) = ui else {
            return Vec::new();
        };
        let view = &target.create_view(&wgpu::TextureViewDescriptor {
            format: Some(self.ui_format),
            ..Default::default()
        });
        let screen = egui_wgpu::ScreenDescriptor {
            size_in_pixels: [width, height],
            pixels_per_point: ui.pixels_per_point,
        };
        let commands =
            self.ui
                .update_buffers(&self.device, &self.queue, encoder, ui.primitives, &screen);
        let mut pass = encoder
            .begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("interface"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Load,
                        store: wgpu::StoreOp::Store,
                    },
                })],
                ..Default::default()
            })
            .forget_lifetime();
        self.ui.render(&mut pass, ui.primitives, &screen);
        commands
    }

    fn update_ui_textures(&mut self, ui: Option<&UiFrame<'_>>) {
        for (id, deltas) in ui.iter().flat_map(|ui| &ui.textures.set) {
            for delta in deltas {
                self.ui
                    .update_texture(&self.device, &self.queue, *id, delta);
            }
        }
    }

    /// Call once the frame using the textures has been submitted.
    fn free_ui_textures(&mut self, ui: Option<&UiFrame<'_>>) {
        for id in ui.iter().flat_map(|ui| &ui.textures.free) {
            self.ui.free_texture(id);
        }
    }

    fn configure_surface(&self) {
        if self.config.width == 0 || self.config.height == 0 {
            return;
        }
        if let Some(surface) = &self.surface {
            surface.configure(&self.device, &self.config);
        }
    }
}

impl fmt::Debug for Renderer {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Renderer")
            .field("adapter", &self.adapter.get_info())
            .field("config", &self.config)
            .field("suspended", &self.surface.is_none())
            .finish_non_exhaustive()
    }
}

fn create_surface(
    instance: &wgpu::Instance,
    window: &Arc<dyn wgpu::WindowHandle>,
) -> Result<wgpu::Surface<'static>> {
    instance
        .create_surface(wgpu::SurfaceTarget::from_window_without_display(
            window.clone(),
        ))
        .context("failed to create the window surface")
}

/// An sRGB colour as the surface `format` expects it: sRGB surfaces take
/// linear values and encode them on write.
fn color(srgb: [f64; 3], format: wgpu::TextureFormat) -> wgpu::Color {
    let [r, g, b] = if format.is_srgb() {
        srgb.map(srgb_to_linear)
    } else {
        srgb
    };
    wgpu::Color { r, g, b, a: 1.0 }
}

pub(crate) fn srgb_to_linear(c: f64) -> f64 {
    if c <= 0.04045 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn auto_tries_modern_apis_before_gl() {
        let attempts = GpuBackend::Auto.attempts();
        assert!(!attempts[0].contains(wgpu::Backends::GL));
        assert_eq!(attempts.last(), Some(&wgpu::Backends::GL));
    }

    #[test]
    fn explicit_backend_has_no_fallback() {
        for backend in [
            GpuBackend::Vulkan,
            GpuBackend::Metal,
            GpuBackend::Dx12,
            GpuBackend::Gl,
        ] {
            assert_eq!(backend.attempts().len(), 1, "{backend:?}");
        }
    }

    #[test]
    fn srgb_to_linear_matches_reference_values() {
        assert_eq!(srgb_to_linear(0.0), 0.0);
        assert!((srgb_to_linear(1.0) - 1.0).abs() < 1e-12);
        assert!((srgb_to_linear(0.5) - 0.214_041).abs() < 1e-6);
    }
}
