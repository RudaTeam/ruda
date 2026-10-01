//! Renderer on top of wgpu.
//!
//! wgpu types never leave this crate: the rest of the engine sees the
//! [`Renderer`], the [`Camera`] and the meshing helpers.

mod camera;
mod mesh;
mod mesher;
mod textures;
mod world_pass;

use std::fmt;
use std::sync::Arc;

use anyhow::{Context as _, Result, bail};
use ruda_core::{BlockPos, ChunkPos, Content};
use tracing::{info, warn};
use wgpu::rwh::{HasDisplayHandle, HasWindowHandle};

pub use camera::{Camera, Frustum};
pub use mesh::{BlockFaces, ChunkMesh, PaddedChunk, Quad, mesh_chunk};
pub use mesher::ChunkMesher;
pub use textures::BlockTextures;

use world_pass::WorldPass;

/// What to draw in a frame.
#[derive(Clone, Copy, Debug)]
pub struct Scene {
    pub camera: Camera,
    /// The block under the crosshair, outlined.
    pub target: Option<BlockPos>,
    /// How far the world is drawn, in blocks; fog hides the edge.
    pub view_distance: f32,
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
    window: Arc<dyn wgpu::WindowHandle>,
    config: wgpu::SurfaceConfiguration,
    /// `None` while suspended: mobile platforms destroy the native window.
    surface: Option<wgpu::Surface<'static>>,
    world: WorldPass,
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
        let (device, queue) = adapter
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
        let renderer = Self {
            instance,
            adapter,
            device,
            queue,
            window,
            config,
            surface: Some(surface),
            world,
        };
        renderer.configure_surface();
        Ok(renderer)
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

    /// Replaces the geometry drawn for a chunk.
    pub fn upload_chunk(&mut self, pos: ChunkPos, mesh: &ChunkMesh) {
        self.world.upload(&self.device, pos, mesh);
    }

    pub fn remove_chunk(&mut self, pos: ChunkPos) {
        self.world.remove(pos);
    }

    /// Chunks with geometry on the GPU.
    pub fn chunk_count(&self) -> usize {
        self.world.chunk_count()
    }

    /// Draws `scene` into an off-screen image instead of the window and
    /// returns its width, height and RGBA pixels.
    pub fn capture(&mut self, scene: &Scene) -> Result<(u32, u32, Vec<u8>)> {
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
            view_formats: &[],
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
        let view = texture.create_view(&wgpu::TextureViewDescriptor::default());
        self.world.draw(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            scene,
            (width, height),
            sky_color(format),
        );
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
        self.queue.submit([encoder.finish()]);

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
        if self.surface.is_none() {
            self.surface = Some(create_surface(&self.instance, &self.window)?);
            self.configure_surface();
        }
        Ok(())
    }

    /// Draws a frame and returns whether it reached the screen: nothing is
    /// presented while the window is hidden, zero-sized or being reconfigured.
    /// `pre_present` runs right before presenting (winit wants
    /// `Window::pre_present_notify` there).
    pub fn render(&mut self, scene: &Scene, pre_present: impl FnOnce()) -> Result<bool> {
        if self.config.width == 0 || self.config.height == 0 {
            return Ok(false);
        }
        let Some(surface) = &self.surface else {
            return Ok(false);
        };

        let (frame, suboptimal) = match surface.get_current_texture() {
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
                self.surface = Some(create_surface(&self.instance, &self.window)?);
                self.configure_surface();
                return Ok(false);
            }
            wgpu::CurrentSurfaceTexture::Validation => {
                bail!("validation error while acquiring a frame")
            }
        };

        let view = frame
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("frame"),
            });
        self.world.draw(
            &self.device,
            &self.queue,
            &mut encoder,
            &view,
            scene,
            (self.config.width, self.config.height),
            sky_color(self.config.format),
        );
        self.queue.submit([encoder.finish()]);

        pre_present();
        self.queue.present(frame);
        if suboptimal {
            self.configure_surface();
        }
        Ok(true)
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

/// Sky blue as sRGB components.
const SKY: [f64; 3] = [0.53, 0.81, 0.92];

fn sky_color(format: wgpu::TextureFormat) -> wgpu::Color {
    // sRGB surfaces take linear values and encode them on write.
    let [r, g, b] = if format.is_srgb() {
        SKY.map(srgb_to_linear)
    } else {
        SKY
    };
    wgpu::Color { r, g, b, a: 1.0 }
}

fn srgb_to_linear(c: f64) -> f64 {
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
