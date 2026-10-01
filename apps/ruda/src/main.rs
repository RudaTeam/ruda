//! Game client: a window with the world of an integrated server.

mod game;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use ruda_render::{GpuBackend, Renderer};
use tracing::{info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, WindowEvent};
use winit::event_loop::{ActiveEventLoop, EventLoop, OwnedDisplayHandle};
use winit::window::{CursorGrabMode, Icon, Window, WindowId};

use crate::game::{Control, Game, GameConfig};

const APP_ICON_PNG: &[u8] = include_bytes!("../../../assets/branding/app-icon.png");

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Graphics API; `auto` prefers Vulkan/Metal/DX12 and falls back to OpenGL.
    #[arg(long, value_enum, default_value_t, env = "RUDA_GPU_BACKEND")]
    gpu_backend: GpuBackendArg,

    /// World seed; a random one if not given.
    #[arg(long)]
    seed: Option<u64>,

    /// How far the world is loaded and drawn, in chunks of 32 blocks.
    #[arg(long, value_name = "CHUNKS", default_value_t = 6, value_parser = clap::value_parser!(i32).range(2..=32))]
    view_distance: i32,

    /// Exit after presenting this many frames (smoke tests, benchmarks).
    #[arg(long, value_name = "N")]
    exit_after_frames: Option<u64>,

    /// With --exit-after-frames, save the last frame to this PNG file.
    #[arg(long, value_name = "PATH", requires = "exit_after_frames")]
    screenshot: Option<PathBuf>,
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
        game: None,
        cursor_grabbed: false,
        last_frame: None,
        frames: 0,
        first_frame_at: None,
        title: TitleStats::default(),
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

/// Window and taskbar icon on Windows and X11. macOS takes the icon from the
/// app bundle instead, and Wayland from the desktop entry.
fn window_icon() -> Result<Icon> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(APP_ICON_PNG)).read_info()?;
    let size = reader
        .output_buffer_size()
        .context("app icon is too large")?;
    let mut rgba = vec![0; size];
    let frame = reader.next_frame(&mut rgba)?;
    anyhow::ensure!(
        frame.color_type == png::ColorType::Rgba && frame.bit_depth == png::BitDepth::Eight,
        "app icon must be 8-bit RGBA"
    );
    rgba.truncate(frame.buffer_size());
    Ok(Icon::from_rgba(rgba, frame.width, frame.height)?)
}

fn save_png(path: &Path, width: u32, height: u32, rgba: &[u8]) -> Result<()> {
    let file = std::io::BufWriter::new(
        std::fs::File::create(path).with_context(|| format!("can't create {}", path.display()))?,
    );
    let mut encoder = png::Encoder::new(file, width, height);
    encoder.set_color(png::ColorType::Rgba);
    encoder.set_depth(png::BitDepth::Eight);
    encoder.write_header()?.write_image_data(rgba)?;
    Ok(())
}

/// Frame counting for the window title.
#[derive(Debug)]
struct TitleStats {
    since: Instant,
    frames: u32,
}

impl Default for TitleStats {
    fn default() -> Self {
        Self {
            since: Instant::now(),
            frames: 0,
        }
    }
}

struct App {
    args: Args,
    display: OwnedDisplayHandle,
    window: Option<Arc<Window>>,
    renderer: Option<Renderer>,
    game: Option<Game>,
    cursor_grabbed: bool,
    last_frame: Option<Instant>,
    /// Frames that actually reached the screen.
    frames: u64,
    first_frame_at: Option<Instant>,
    title: TitleStats,
    occluded: bool,
    /// Minimized on Windows: the surface cannot be configured at 0×0.
    zero_sized: bool,
    /// First fatal error; `main` returns it once the event loop has exited.
    error: Option<anyhow::Error>,
}

impl App {
    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let icon = window_icon()
            .inspect_err(|error| warn!("no window icon: {error:#}"))
            .ok();
        let attributes = Window::default_attributes()
            .with_title("Ruda")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_window_icon(icon);
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create the window")?,
        );
        let size = window.inner_size();
        let mut renderer = pollster::block_on(Renderer::new(
            self.display.clone(),
            window.clone(),
            size.width,
            size.height,
            self.args.gpu_backend.into(),
        ))?;

        let seed = self.args.seed.unwrap_or_else(random_seed);
        let config = GameConfig {
            seed,
            view_distance: self.args.view_distance,
        };
        self.game = Some(Game::start(config, &mut renderer)?);

        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        Ok(())
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let (Some(window), Some(renderer), Some(game)) =
            (&self.window, &mut self.renderer, &mut self.game)
        else {
            return Ok(());
        };
        let _span = tracing::info_span!("frame").entered();

        let now = Instant::now();
        // A long hitch (dragging the window, a breakpoint) shouldn't teleport.
        let dt = self
            .last_frame
            .map_or(0.0, |last| (now - last).as_secs_f64().min(0.1));
        self.last_frame = Some(now);
        let control = game.update(dt, self.cursor_grabbed, renderer)?;

        let scene = game.scene();
        if renderer.render(&scene, || window.pre_present_notify())? {
            self.frames += 1;
            self.title.frames += 1;
            self.first_frame_at.get_or_insert(now);
            #[cfg(feature = "tracy")]
            tracing_tracy::client::frame_mark();
        }

        let elapsed = now - self.title.since;
        if elapsed >= Duration::from_millis(500) {
            let fps = f64::from(self.title.frames) / elapsed.as_secs_f64();
            window.set_title(&format!("Ruda — {fps:.0} FPS · {}", game.status()));
            self.title = TitleStats::default();
        }

        if self
            .args
            .exit_after_frames
            .is_some_and(|n| self.frames >= n)
        {
            if let Some(path) = &self.args.screenshot {
                let (width, height, pixels) = renderer.capture(&scene)?;
                save_png(path, width, height, &pixels)?;
                info!(path = %path.display(), "saved a screenshot");
            }
            let seconds = self
                .first_frame_at
                .map_or(0.0, |at| at.elapsed().as_secs_f64());
            let fps = if seconds > 0.0 {
                self.frames.saturating_sub(1) as f64 / seconds
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

        if control == Control::ReleaseCursor {
            self.grab_cursor(false);
        }
        Ok(())
    }

    fn grab_cursor(&mut self, grab: bool) {
        let Some(window) = &self.window else {
            return;
        };
        if grab {
            // Locked keeps the cursor in place; not every platform has it.
            let grabbed = window
                .set_cursor_grab(CursorGrabMode::Locked)
                .or_else(|_| window.set_cursor_grab(CursorGrabMode::Confined));
            if let Err(error) = grabbed {
                warn!("can't capture the mouse: {error}");
                return;
            }
        } else if let Err(error) = window.set_cursor_grab(CursorGrabMode::None) {
            warn!("can't release the mouse: {error}");
        }
        window.set_cursor_visible(!grab);
        self.cursor_grabbed = grab;
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
        if let Some(game) = &mut self.game {
            game.input.window_event(&event);
        }
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
            // The first click only captures the mouse.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } if !self.cursor_grabbed => {
                self.grab_cursor(true);
                if let Some(game) = &mut self.game {
                    game.input.take_pressed();
                }
            }
            WindowEvent::Focused(false) => self.grab_cursor(false),
            WindowEvent::RedrawRequested => {
                if let Err(error) = self.redraw(event_loop) {
                    self.fail(event_loop, error);
                }
            }
            _ => {}
        }
    }

    fn device_event(&mut self, _event_loop: &ActiveEventLoop, _id: DeviceId, event: DeviceEvent) {
        if self.cursor_grabbed
            && let Some(game) = &mut self.game
        {
            game.input.device_event(&event);
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        if let Some(game) = self.game.take() {
            game.shutdown();
        }
    }
}

fn random_seed() -> u64 {
    SystemTime::now()
        .duration_since(SystemTime::UNIX_EPOCH)
        .map_or(0, |time| time.as_nanos() as u64)
}

#[cfg(test)]
mod tests {
    #[test]
    fn app_icon_decodes() {
        super::window_icon().unwrap();
    }
}
