//! Game client. For now it opens a window and clears it every frame.

use std::sync::Arc;
use std::time::Instant;

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use ruda_render::{GpuBackend, Renderer};
use tracing::info;
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::WindowEvent;
use winit::event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle};
use winit::window::{Window, WindowId};

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Graphics API; `auto` prefers Vulkan/Metal/DX12 and falls back to OpenGL.
    #[arg(long, value_enum, default_value_t, env = "RUDA_GPU_BACKEND")]
    gpu_backend: GpuBackendArg,

    /// Exit after presenting this many frames (smoke tests, benchmarks).
    #[arg(long, value_name = "N")]
    exit_after_frames: Option<u64>,
}

#[derive(Clone, Copy, Debug, Default, ValueEnum)]
enum GpuBackendArg {
    #[default]
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl From<GpuBackendArg> for GpuBackend {
    fn from(arg: GpuBackendArg) -> Self {
        match arg {
            GpuBackendArg::Auto => Self::Auto,
            GpuBackendArg::Vulkan => Self::Vulkan,
            GpuBackendArg::Metal => Self::Metal,
            GpuBackendArg::Dx12 => Self::Dx12,
            GpuBackendArg::Gl => Self::Gl,
        }
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();
    info!(version = env!("CARGO_PKG_VERSION"), "starting ruda");

    let event_loop = EventLoop::new().context("failed to create the event loop")?;
    let mut app = App {
        display: event_loop.owned_display_handle(),
        args,
        window: None,
        renderer: None,
        frames: 0,
        first_frame_at: None,
        occluded: false,
        zero_sized: false,
        error: None,
    };
    event_loop.run_app(&mut app).context("event loop failed")?;
    app.error.map_or(Ok(()), Err)
}

fn init_tracing() {
    use tracing_subscriber::prelude::*;

    // wgpu is chatty at `info`.
    let filter = tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| {
        tracing_subscriber::EnvFilter::new("info,wgpu_core=warn,wgpu_hal=warn,naga=warn")
    });
    let registry = tracing_subscriber::registry()
        .with(filter)
        .with(tracing_subscriber::fmt::layer());
    #[cfg(feature = "tracy")]
    let registry = registry.with(tracing_tracy::TracyLayer::default());
    registry.init();
}

struct App {
    args: Args,
    display: OwnedDisplayHandle,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    /// Frames that actually reached the screen.
    frames: u64,
    first_frame_at: Option<Instant>,
    occluded: bool,
    /// Minimized on Windows: the surface cannot be configured at 0×0.
    zero_sized: bool,
    /// First fatal error; `main` returns it once the event loop has exited.
    error: Option<anyhow::Error>,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let attributes = Window::default_attributes()
            .with_title("Ruda")
            .with_inner_size(LogicalSize::new(1280, 720));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create the window")?,
        );
        let size = window.inner_size();
        let renderer = pollster::block_on(Renderer::new(
            self.display.clone(),
            window.clone(),
            size.width,
            size.height,
            self.args.gpu_backend.into(),
        ))?;

        window.set_title(&format!("Ruda — {}", renderer.adapter_summary()));
        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        Ok(())
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let (Some(window), Some(renderer)) = (&self.window, &mut self.renderer) else {
            return Ok(());
        };
        let _span = tracing::info_span!("frame").entered();
        if renderer.render(|| window.pre_present_notify())? {
            self.frames += 1;
            self.first_frame_at.get_or_insert_with(Instant::now);
            #[cfg(feature = "tracy")]
            tracing_tracy::client::frame_mark();
        }

        if self
            .args
            .exit_after_frames
            .is_some_and(|n| self.frames >= n)
        {
            let elapsed = self
                .first_frame_at
                .map_or(0.0, |at| at.elapsed().as_secs_f64());
            let fps = if elapsed > 0.0 {
                self.frames.saturating_sub(1) as f64 / elapsed
            } else {
                0.0
            };
            info!(
                frames = self.frames,
                fps = %format_args!("{fps:.1}"),
                "frame limit reached, exiting"
            );
            event_loop.exit();
        } else if !self.occluded && !self.zero_sized {
            window.request_redraw();
        }
        Ok(())
    }

    fn request_redraw(&self) {
        if let Some(window) = &self.window {
            window.request_redraw();
        }
    }

    fn fail(&mut self, event_loop: &ActiveEventLoop, error: anyhow::Error) {
        self.error.get_or_insert(error);
        event_loop.exit();
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        let result = match &mut self.renderer {
            Some(renderer) => renderer.resume(),
            None => self.init(event_loop),
        };
        if let Err(error) = result {
            self.fail(event_loop, error);
        }
    }

    fn suspended(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(renderer) = &mut self.renderer {
            renderer.suspend();
        }
    }

    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => {
                self.zero_sized = size.width == 0 || size.height == 0;
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                self.request_redraw();
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    self.request_redraw();
                }
            }
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw(event_loop) {
                    self.fail(event_loop, error);
                }
            }
            _ => {}
        }
    }
}
