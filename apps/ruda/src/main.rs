//! Game client: a window with the menus and the world of an integrated server.

mod game;
mod interface;
mod settings;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use ruda_render::{Backdrop, GpuBackend, Renderer};
use ruda_ui::{GpuApi, I18n, Language, Menu, MenuAction, MenuContext, Screen, Settings};
use tracing::{info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Icon, Window, WindowId};

use crate::game::{Control, Game, GameConfig, MAX_VIEW_DISTANCE};
use crate::interface::Interface;

const APP_ICON_PNG: &[u8] = include_bytes!("../../../assets/branding/app-icon.png");

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Start a singleplayer world right away instead of showing the main menu.
    #[arg(long)]
    singleplayer: bool,

    /// World seed; a random one if not given.
    #[arg(long)]
    seed: Option<u64>,

    /// Graphics API instead of the one in the settings; `auto` prefers
    /// Vulkan/Metal/DX12 and falls back to OpenGL.
    #[arg(long, value_enum, env = "RUDA_GPU_BACKEND")]
    gpu_backend: Option<GpuBackendArg>,

    /// How far the world is loaded and drawn, in chunks of 32 blocks, instead
    /// of the distance in the settings.
    #[arg(long, value_name = "CHUNKS", value_parser = clap::value_parser!(u8).range(2..=i64::from(MAX_VIEW_DISTANCE)))]
    view_distance: Option<u8>,

    /// Exit after presenting this many frames (smoke tests, benchmarks).
    #[arg(long, value_name = "N")]
    exit_after_frames: Option<u64>,

    /// With --exit-after-frames, save the last frame to this PNG file.
    #[arg(long, value_name = "PATH", requires = "exit_after_frames")]
    screenshot: Option<PathBuf>,
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum GpuBackendArg {
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

fn gpu_backend(api: GpuApi) -> GpuBackend {
    match api {
        GpuApi::Auto => GpuBackend::Auto,
        GpuApi::Vulkan => GpuBackend::Vulkan,
        GpuApi::Metal => GpuBackend::Metal,
        GpuApi::Dx12 => GpuBackend::Dx12,
        GpuApi::Gl => GpuBackend::Gl,
    }
}

fn main() -> Result<()> {
    let args = Args::parse();
    init_tracing();
    info!(version = env!("CARGO_PKG_VERSION"), "starting ruda");

    let event_loop = EventLoop::new().context("failed to create the event loop")?;
    let mut app = App::new(args, event_loop.owned_display_handle());
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

/// Width, height and 8-bit RGBA pixels of a PNG image.
fn decode_png(png: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let mut reader = png::Decoder::new(std::io::Cursor::new(png)).read_info()?;
    let size = reader.output_buffer_size().context("image is too large")?;
    let mut rgba = vec![0; size];
    let frame = reader.next_frame(&mut rgba)?;
    anyhow::ensure!(
        frame.color_type == png::ColorType::Rgba && frame.bit_depth == png::BitDepth::Eight,
        "image must be 8-bit RGBA"
    );
    rgba.truncate(frame.buffer_size());
    Ok((frame.width, frame.height, rgba))
}

/// Window and taskbar icon on Windows and X11. macOS takes the icon from the
/// app bundle instead, and Wayland from the desktop entry.
fn window_icon() -> Result<Icon> {
    let (width, height, rgba) = decode_png(APP_ICON_PNG)?;
    Ok(Icon::from_rgba(rgba, width, height)?)
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

fn fullscreen(on: bool) -> Option<Fullscreen> {
    on.then_some(Fullscreen::Borderless(None))
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
    interface: Option<Interface>,
    game: Option<Game>,
    /// The open menu; `None` while playing.
    menu: Option<Menu>,
    settings: Settings,
    /// The settings in effect, to notice when the menu changes them.
    applied: Settings,
    /// The settings on disk, to save only when something changed.
    saved: Settings,
    settings_path: Option<PathBuf>,
    i18n: I18n,
    system_language: Language,
    cursor_grabbed: bool,
    last_frame: Option<Instant>,
    /// Frames that actually reached the screen.
    frames: u64,
    first_frame_at: Option<Instant>,
    title: TitleStats,
    occluded: bool,
    /// Minimized on Windows: the surface cannot be configured at 0×0.
    zero_sized: bool,
    /// When a menu with nothing going on next needs drawing.
    repaint_at: Option<Instant>,
    /// First fatal error; `main` returns it once the event loop has exited.
    error: Option<anyhow::Error>,
}

impl App {
    fn new(args: Args, display: OwnedDisplayHandle) -> Self {
        let settings_path = settings::path();
        let settings = settings_path
            .as_deref()
            .map(settings::load)
            .unwrap_or_default();
        let system_language = sys_locale::get_locale()
            .map_or(Language::English, |locale| Language::from_locale(&locale));
        Self {
            args,
            display,
            window: None,
            renderer: None,
            interface: None,
            game: None,
            menu: None,
            i18n: I18n::new(settings.language.unwrap_or(system_language)),
            applied: settings.clone(),
            saved: settings.clone(),
            settings,
            settings_path,
            system_language,
            cursor_grabbed: false,
            last_frame: None,
            frames: 0,
            first_frame_at: None,
            title: TitleStats::default(),
            occluded: false,
            zero_sized: false,
            repaint_at: None,
            error: None,
        }
    }

    fn init(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let graphics = &self.settings.graphics;
        let icon = window_icon()
            .inspect_err(|error| warn!("no window icon: {error:#}"))
            .ok();
        let attributes = Window::default_attributes()
            .with_title("Ruda")
            .with_inner_size(LogicalSize::new(1280, 720))
            .with_window_icon(icon)
            .with_fullscreen(fullscreen(graphics.fullscreen));
        let window = Arc::new(
            event_loop
                .create_window(attributes)
                .context("failed to create the window")?,
        );
        let size = window.inner_size();
        let new_renderer = |backend| {
            pollster::block_on(Renderer::new(
                self.display.clone(),
                window.clone(),
                size.width,
                size.height,
                backend,
            ))
        };
        let mut renderer = match self.args.gpu_backend {
            Some(arg) => new_renderer(arg.into())?,
            // A graphics API picked in the settings may have stopped working;
            // don't lock the player out of the menu where it can be changed.
            None => match new_renderer(gpu_backend(graphics.gpu_api)) {
                Err(error) if graphics.gpu_api != GpuApi::Auto => {
                    warn!("{error:#}; trying the other graphics APIs");
                    new_renderer(GpuBackend::Auto)?
                }
                renderer => renderer?,
            },
        };
        if !graphics.vsync {
            renderer.set_vsync(false);
        }

        self.interface = Some(Interface::new(&window, renderer.max_texture_side()));
        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        if self.args.singleplayer {
            self.start_game()?;
        } else {
            self.menu = Some(Menu::new(Screen::Main));
        }
        Ok(())
    }

    fn start_game(&mut self) -> Result<()> {
        let Some(renderer) = &mut self.renderer else {
            return Ok(());
        };
        let graphics = &self.settings.graphics;
        let config = GameConfig {
            seed: self.args.seed.unwrap_or_else(random_seed),
            view_distance: self.args.view_distance.unwrap_or(graphics.view_distance),
            fov: f32::from(graphics.fov),
        };
        self.game = Some(Game::start(config, renderer)?);
        self.resume();
        Ok(())
    }

    fn quit_to_title(&mut self) {
        if let Some(game) = self.game.take() {
            game.shutdown();
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.clear_chunks();
        }
        self.menu = Some(Menu::new(Screen::Main));
    }

    /// Opens the pause menu over a running game.
    fn pause(&mut self) {
        let Some(game) = &mut self.game else {
            return;
        };
        if self.menu.is_none() {
            game.input.clear();
            self.menu = Some(Menu::new(Screen::Paused));
            self.grab_cursor(false);
        }
    }

    fn resume(&mut self) {
        if let Some(game) = &mut self.game {
            game.input.clear();
        }
        self.menu = None;
        self.grab_cursor(true);
    }

    fn on_menu_action(&mut self, event_loop: &ActiveEventLoop, action: MenuAction) -> Result<()> {
        match action {
            MenuAction::StartSingleplayer => self.start_game()?,
            MenuAction::Resume => self.resume(),
            MenuAction::QuitToTitle => self.quit_to_title(),
            MenuAction::Exit => event_loop.exit(),
            MenuAction::SettingsClosed => self.save_settings(),
        }
        self.request_redraw();
        Ok(())
    }

    /// Puts changes made in the settings menu into effect.
    fn apply_settings(&mut self) {
        if self.settings == self.applied {
            return;
        }
        let (new, old) = (&self.settings.graphics, &self.applied.graphics);
        if new.vsync != old.vsync
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_vsync(new.vsync);
        }
        if new.fullscreen != old.fullscreen
            && let Some(window) = &self.window
        {
            window.set_fullscreen(fullscreen(new.fullscreen));
        }
        if let Some(game) = &mut self.game {
            if new.view_distance != old.view_distance {
                game.set_view_distance(new.view_distance);
            }
            if new.fov != old.fov {
                game.set_fov(f32::from(new.fov));
            }
        }
        self.i18n
            .set_language(self.settings.language.unwrap_or(self.system_language));
        self.applied = self.settings.clone();
    }

    fn save_settings(&mut self) {
        let Some(path) = &self.settings_path else {
            return;
        };
        if self.settings == self.saved {
            return;
        }
        match settings::save(path, &self.settings) {
            Ok(()) => {
                info!(path = %path.display(), "saved the settings");
                self.saved = self.settings.clone();
            }
            Err(error) => warn!("{error:#}"),
        }
    }

    fn redraw(&mut self, event_loop: &ActiveEventLoop) -> Result<()> {
        let _span = tracing::info_span!("frame").entered();

        let now = Instant::now();
        // A long hitch (dragging the window, a breakpoint) shouldn't teleport.
        let dt = self
            .last_frame
            .map_or(0.0, |last| (now - last).as_secs_f64().min(0.1));
        self.last_frame = Some(now);
        if let (Some(game), Some(renderer)) = (&mut self.game, &mut self.renderer)
            && game.update(dt, self.cursor_grabbed, renderer)? == Control::Pause
        {
            self.pause();
        }

        let (Some(window), Some(renderer), Some(interface)) =
            (&self.window, &mut self.renderer, &mut self.interface)
        else {
            return Ok(());
        };
        let mut action = None;
        let painted = match &mut self.menu {
            Some(menu) => {
                let logo = interface.logo.clone();
                let context = MenuContext {
                    i18n: &self.i18n,
                    logo: logo.as_ref(),
                    version: env!("CARGO_PKG_VERSION"),
                    system_language: self.system_language,
                };
                let settings = &mut self.settings;
                Some(interface.run(window, |ui| {
                    if let Some(clicked) = menu.show(ui, context, settings) {
                        action = Some(clicked);
                    }
                }))
            }
            None => {
                interface.skip_frame(window);
                None
            }
        };
        let ui = painted.as_ref().map(|painted| painted.frame());
        let scene = self.game.as_ref().map(Game::scene);
        let backdrop = match &scene {
            Some(scene) => Backdrop::World(scene),
            None => {
                let color = ruda_ui::BACKGROUND;
                Backdrop::Color([color.r(), color.g(), color.b()])
            }
        };
        if renderer.render(backdrop, ui.as_ref(), || window.pre_present_notify())? {
            self.frames += 1;
            self.title.frames += 1;
            self.first_frame_at.get_or_insert(now);
            #[cfg(feature = "tracy")]
            tracing_tracy::client::frame_mark();
        }

        let elapsed = now - self.title.since;
        if elapsed >= Duration::from_millis(500) {
            let fps = f64::from(self.title.frames) / elapsed.as_secs_f64();
            let status = self.game.as_ref().map(Game::status).unwrap_or_default();
            window.set_title(&format!("Ruda — {fps:.0} FPS · {status}"));
            self.title = TitleStats::default();
        }

        // Menus are drawn only when something changes; the game every frame.
        let repaint_after = painted
            .as_ref()
            .filter(|_| self.game.is_none() && self.args.exit_after_frames.is_none())
            .map_or(Duration::ZERO, |painted| painted.repaint_after);
        if self
            .args
            .exit_after_frames
            .is_some_and(|n| self.frames >= n)
        {
            if let Some(path) = &self.args.screenshot {
                let (width, height, pixels) = renderer.capture(backdrop, ui.as_ref())?;
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
        } else if self.occluded || self.zero_sized {
            self.repaint_at = None;
        } else if repaint_after.is_zero() {
            window.request_redraw();
        } else {
            self.repaint_at = now.checked_add(repaint_after);
        }

        self.apply_settings();
        if let Some(action) = action {
            self.on_menu_action(event_loop, action)?;
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

    fn request_redraw(&mut self) {
        self.repaint_at = None;
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
        // The interface keeps track of the window even while playing, but
        // only menus act on input. It asks to redraw after every redraw, so
        // it doesn't see those.
        if let (Some(interface), Some(window)) = (&mut self.interface, &self.window)
            && event != WindowEvent::RedrawRequested
            && interface.window_event(window, &event)
            && self.menu.is_some()
        {
            self.request_redraw();
        }
        if self.menu.is_none()
            && let Some(game) = &mut self.game
        {
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
            // Escape in a menu goes back; while playing the game sees it.
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(KeyCode::Escape),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } => {
                if let Some(action) = self.menu.as_mut().and_then(Menu::back)
                    && let Err(error) = self.on_menu_action(event_loop, action)
                {
                    self.fail(event_loop, error);
                }
            }
            // Playing without the mouse, e.g. after the system took it away:
            // a click only captures it again.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                ..
            } if self.menu.is_none() && !self.cursor_grabbed => {
                self.grab_cursor(true);
                if let Some(game) = &mut self.game {
                    game.input.clear();
                }
            }
            WindowEvent::Focused(false) => self.pause(),
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
            && self.menu.is_none()
            && let Some(game) = &mut self.game
        {
            game.input.device_event(&event);
        }
    }

    fn about_to_wait(&mut self, event_loop: &ActiveEventLoop) {
        match self.repaint_at {
            Some(at) if at <= Instant::now() => {
                self.request_redraw();
                event_loop.set_control_flow(ControlFlow::Wait);
            }
            Some(at) => event_loop.set_control_flow(ControlFlow::WaitUntil(at)),
            None => event_loop.set_control_flow(ControlFlow::Wait),
        }
    }

    fn exiting(&mut self, _event_loop: &ActiveEventLoop) {
        self.save_settings();
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
    fn images_decode() {
        super::window_icon().unwrap();
        super::decode_png(super::interface::LOGO_PNG).unwrap();
    }
}
