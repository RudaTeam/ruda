//! Game client: a window with the menus and the world of an integrated server.

mod benchmark;
mod game;
mod icons;
mod interface;
mod settings;

use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant, SystemTime};

use anyhow::{Context as _, Result};
use clap::{Parser, ValueEnum};
use ruda_core::BlockId;
use ruda_input::{Action, Bindings, Button};
use ruda_render::{Backdrop, GpuBackend, Renderer};
use ruda_ui::{
    ControlAction, Controls, FpsLimit, GpuApi, HotbarSlot, Hud, HudContext, I18n, InventoryContext,
    Language, Menu, MenuAction, MenuContext, Screen, Settings,
};
use ruda_ui::{Lighting, Preset, Shadows};
use tracing::{info, warn};
use winit::application::ApplicationHandler;
use winit::dpi::LogicalSize;
use winit::event::{DeviceEvent, DeviceId, ElementState, KeyEvent, MouseButton, WindowEvent};
use winit::event_loop::{ActiveEventLoop, ControlFlow, EventLoop, OwnedDisplayHandle};
use winit::keyboard::{KeyCode, PhysicalKey};
use winit::window::{CursorGrabMode, Fullscreen, Icon, Window, WindowId};

use crate::benchmark::Benchmark;
use crate::game::{CameraStart, Control, Game, GameConfig, MAX_VIEW_DISTANCE};
use crate::interface::Interface;

const APP_ICON_PNG: &[u8] = include_bytes!("../../../assets/branding/app-icon.png");

/// The least time between frames of the world behind the title menu.
const TITLE_FRAME: Duration = Duration::from_millis(33);
/// The startup screen stays at least this long, so the logo is not a
/// flicker when the world loads quickly.
const SPLASH_AT_LEAST: Duration = Duration::from_millis(1500);

#[derive(Debug, Parser)]
#[command(version, about)]
struct Args {
    /// Start a singleplayer world right away instead of showing the main menu.
    #[arg(long)]
    singleplayer: bool,

    /// World seed; a random one if not given.
    #[arg(long)]
    seed: Option<u64>,

    /// Start the world at this time of day, in ticks: 0 is sunrise, 6000
    /// noon, 12000 sunset and 18000 midnight.
    #[arg(long, value_name = "TICKS")]
    time: Option<u64>,

    /// Start the camera at `x,y,z` or `x,y,z,yaw,pitch` (degrees) instead of
    /// the spawn point, the player flying so it stays there.
    #[arg(long, value_name = "X,Y,Z[,YAW,PITCH]", allow_hyphen_values = true)]
    camera: Option<CameraStart>,

    /// Graphics API instead of the one in the settings; `auto` prefers
    /// Vulkan/Metal/DX12 and falls back to OpenGL.
    #[arg(long, value_enum, env = "RUDA_GPU_BACKEND")]
    gpu_backend: Option<GpuBackendArg>,

    /// How far the world is loaded and drawn, in chunks of 32 blocks, instead
    /// of the distance in the settings.
    #[arg(long, value_name = "CHUNKS", value_parser = clap::value_parser!(u8).range(2..=i64::from(MAX_VIEW_DISTANCE)))]
    view_distance: Option<u8>,

    /// How far simplified far-away terrain reaches, in blocks (0 for none),
    /// instead of the distance in the settings.
    #[arg(long, value_name = "BLOCKS")]
    lod_distance: Option<u16>,

    /// Exit after presenting this many frames (smoke tests, benchmarks);
    /// those of the loading screen don't count.
    #[arg(long, value_name = "N")]
    exit_after_frames: Option<u64>,

    /// With --exit-after-frames or --benchmark, save the last frame to this
    /// PNG file.
    #[arg(long, value_name = "PATH")]
    screenshot: Option<PathBuf>,

    /// Start a singleplayer world, wait until it has loaded, measure frame
    /// times for this many seconds, print them and exit.
    #[arg(long, value_name = "SECONDS", conflicts_with = "exit_after_frames")]
    benchmark: Option<f64>,

    /// Draw as many frames as possible, whatever the settings say.
    #[arg(long)]
    no_vsync: bool,

    /// How the sun casts shadows, whatever the settings say.
    #[arg(long, value_enum)]
    shadows: Option<ShadowsArg>,

    /// Draw no clouds, whatever the settings say.
    #[arg(long)]
    no_clouds: bool,

    /// Start from this graphics preset for this run; the saved settings
    /// stay as they are unless changed in the menu.
    #[arg(long, value_enum)]
    preset: Option<PresetArg>,

    /// With --benchmark, draw off-screen instead of in a window: nothing can
    /// hide or pause it, and no display is needed.
    #[arg(long, requires = "benchmark")]
    headless: bool,

    /// Size of the off-screen frame for --headless, as WIDTHxHEIGHT.
    #[arg(long, default_value = "2560x1440", value_parser = parse_size)]
    size: (u32, u32),
}

fn parse_size(text: &str) -> Result<(u32, u32), String> {
    let (width, height) = text
        .split_once('x')
        .ok_or_else(|| format!("expected WIDTHxHEIGHT, got {text:?}"))?;
    let number = |text: &str| {
        text.parse::<u32>()
            .ok()
            .filter(|&n| n > 0)
            .ok_or_else(|| format!("{text:?} is not a size in pixels"))
    };
    Ok((number(width)?, number(height)?))
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum ShadowsArg {
    Off,
    Standard,
    Rays,
}

impl From<ShadowsArg> for Shadows {
    fn from(arg: ShadowsArg) -> Self {
        match arg {
            ShadowsArg::Off => Shadows::Off,
            ShadowsArg::Standard => Shadows::Standard,
            ShadowsArg::Rays => Shadows::Rays,
        }
    }
}

#[derive(Clone, Copy, Debug, ValueEnum)]
enum PresetArg {
    Standard,
    High,
    Ultra,
}

impl From<PresetArg> for Preset {
    fn from(arg: PresetArg) -> Self {
        match arg {
            PresetArg::Standard => Preset::Standard,
            PresetArg::High => Preset::High,
            PresetArg::Ultra => Preset::Ultra,
        }
    }
}

/// The saved settings, with the preset of the command line applied.
fn load_settings(path: Option<&Path>, args: &Args) -> Settings {
    let mut settings: Settings = path.map(settings::load).unwrap_or_default();
    // Buttons the input has no name for, or that belong to the hotbar and
    // the pause, are not what the settings can hold.
    settings
        .controls
        .retain_keys(|name| Button::from_name(name).is_some_and(|button| !is_reserved(button)));
    if let Some(preset) = args.preset {
        Preset::from(preset).apply(&mut settings.graphics);
    }
    settings
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

    if args.headless {
        return run_headless(&args);
    }
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

/// Width, height and 8-bit RGBA pixels of a PNG image of any colour type:
/// palettes and grey are expanded, and what has no transparency is made
/// opaque.
fn decode_png(png: &[u8]) -> Result<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(png));
    decoder.set_transformations(png::Transformations::normalize_to_color8());
    let mut reader = decoder.read_info()?;
    let size = reader.output_buffer_size().context("image is too large")?;
    let mut pixels = vec![0; size];
    let frame = reader.next_frame(&mut pixels)?;
    anyhow::ensure!(
        frame.bit_depth == png::BitDepth::Eight,
        "image must have 8 bits a channel"
    );
    let pixels = &pixels[..frame.buffer_size()];
    let rgba = match frame.color_type {
        png::ColorType::Rgba => pixels.to_vec(),
        png::ColorType::Rgb => pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[grey, alpha]| [grey, grey, grey, alpha])
            .collect(),
        png::ColorType::Grayscale => pixels
            .iter()
            .flat_map(|&grey| [grey, grey, grey, 255])
            .collect(),
        png::ColorType::Indexed => anyhow::bail!("the palette was not expanded"),
    };
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
    /// When the startup screen came up.
    splash_since: Option<Instant>,
    hud: Hud,
    /// What each cell of the hotbar of the running game holds.
    hotbar: Vec<Option<HotbarSlot>>,
    /// Every block the inventory offers, and which block each is.
    catalog: Vec<HotbarSlot>,
    catalog_blocks: Vec<BlockId>,
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
    /// Frames that actually reached the screen, but for the loading screen.
    frames: u64,
    first_frame_at: Option<Instant>,
    title: TitleStats,
    occluded: bool,
    /// Minimized on Windows: the surface cannot be configured at 0×0.
    zero_sized: bool,
    /// When a menu with nothing going on next needs drawing.
    repaint_at: Option<Instant>,
    benchmark: Option<Benchmark>,
    /// First fatal error; `main` returns it once the event loop has exited.
    error: Option<anyhow::Error>,
}

impl App {
    fn new(args: Args, display: OwnedDisplayHandle) -> Self {
        let settings_path = settings::path();
        let settings = load_settings(settings_path.as_deref(), &args);
        let system_language = sys_locale::get_locale()
            .map_or(Language::English, |locale| Language::from_locale(&locale));
        Self {
            display,
            window: None,
            renderer: None,
            interface: None,
            game: None,
            menu: None,
            splash_since: None,
            hud: Hud::default(),
            hotbar: Vec::new(),
            catalog: Vec::new(),
            catalog_blocks: Vec::new(),
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
            benchmark: args
                .benchmark
                .map(|seconds| Benchmark::new(Duration::from_secs_f64(seconds))),
            args,
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
        renderer.set_vsync(self.vsync());
        renderer.set_shadows(shadows(&self.args, graphics.shadows));
        renderer.set_lighting(lighting(graphics.lighting));

        let mut interface = Interface::new(&window, renderer.max_texture_side());
        interface.set_scale(&window, self.settings.appearance.scale);
        self.interface = Some(interface);
        window.request_redraw();
        self.window = Some(window);
        self.renderer = Some(renderer);
        if self.args.singleplayer || self.benchmark.is_some() {
            self.start_game()?;
        } else {
            self.menu = Some(Menu::new(Screen::Main));
            self.start_title_world();
            // The world behind the menu loads out of sight, behind the logo.
            if self.game.is_some() {
                self.menu = Some(Menu::new(Screen::Splash));
                self.splash_since = Some(Instant::now());
            }
        }
        Ok(())
    }

    /// Whether frames wait for the display's refresh.
    fn vsync(&self) -> bool {
        self.settings.graphics.fps_limit == FpsLimit::Display && !self.args.no_vsync
    }

    /// The least time from one frame to the next, if frames are limited to
    /// a number a second.
    fn frame_interval(&self) -> Option<Duration> {
        let limit = if self.args.no_vsync {
            None
        } else {
            let fps = self.settings.graphics.fps_limit.fps();
            fps.map(|fps| Duration::from_secs_f64(1.0 / f64::from(fps)))
        };
        // Nobody plays behind the title menu: that much is enough.
        if self.game.as_ref().is_some_and(Game::is_panorama) {
            return Some(limit.map_or(TITLE_FRAME, |limit| limit.max(TITLE_FRAME)));
        }
        limit
    }

    /// Whether there are clouds: as the settings say, unless the command
    /// line says otherwise.
    fn clouds(&self) -> bool {
        clouds(&self.args, self.settings.graphics.clouds)
    }

    /// What a game starts with: the settings, unless the command line says
    /// otherwise.
    fn game_config(&self) -> GameConfig {
        let graphics = &self.settings.graphics;
        GameConfig {
            seed: self.args.seed.unwrap_or_else(random_seed),
            view_distance: self.args.view_distance.unwrap_or(graphics.view_distance),
            fov: f32::from(graphics.fov),
            camera: self.args.camera,
            time: self.args.time,
            lod_distance: self.args.lod_distance.unwrap_or(graphics.lod_distance),
            clouds: self.clouds(),
            auto_jump: self.settings.controls.auto_jump,
            mouse_sensitivity: self.settings.controls.mouse_sensitivity,
            view_bobbing: graphics.view_bobbing,
            panorama: false,
        }
    }

    fn start_game(&mut self) -> Result<()> {
        self.stop_title_world();
        let config = self.game_config();
        let Some(renderer) = &mut self.renderer else {
            return Ok(());
        };
        let mut game = Game::start(config, renderer)?;
        self.hud = Hud::default();
        game.input.set_bindings(bindings(&self.settings.controls));
        if let Some(interface) = &self.interface {
            // Every picture is made once, for the catalog; the hotbar shows
            // the same ones.
            self.catalog_blocks = game.placeable_blocks();
            self.catalog = interface.block_slots(&game, &self.catalog_blocks);
            self.hotbar = game
                .hotbar()
                .iter()
                .map(|block| {
                    let at = self
                        .catalog_blocks
                        .iter()
                        .position(|other| Some(*other) == *block);
                    at.map(|at| self.catalog[at].clone())
                })
                .collect();
        }
        self.game = Some(game);
        // The world shows once the area around the player has loaded.
        self.menu = Some(Menu::new(Screen::Loading));
        Ok(())
    }

    /// Whether a world should drift behind the menu: when the settings want
    /// one and the title menu is up with no game running.
    fn title_world_wanted(&self) -> bool {
        self.settings.graphics.menu_world
            && !self.args.singleplayer
            && self.benchmark.is_none()
            && self.game.is_none()
            && self.menu.is_some_and(|menu| {
                matches!(
                    menu.screen,
                    Screen::Main | Screen::Settings { from_game: false }
                )
            })
    }

    /// Starts the world behind the title menu. Without it the menu shows a
    /// picture of the same world, so failing to start it only costs the
    /// motion.
    fn start_title_world(&mut self) {
        if !self.title_world_wanted() {
            return;
        }
        let config = self.game_config().for_panorama();
        let Some(renderer) = &mut self.renderer else {
            return;
        };
        match Game::start(config, renderer) {
            Ok(game) => self.game = Some(game),
            Err(error) => warn!("no world behind the menu: {error:#}"),
        }
    }

    /// Stops the world behind the title menu, if it is running.
    fn stop_title_world(&mut self) {
        if self.game.as_ref().is_some_and(Game::is_panorama)
            && let Some(game) = self.game.take()
        {
            game.shutdown();
            if let Some(renderer) = &mut self.renderer {
                renderer.clear_chunks();
            }
        }
    }

    /// Whether the loading screen is up.
    fn loading(&self) -> bool {
        self.menu.is_some_and(|menu| menu.screen == Screen::Loading)
    }

    /// Whether the startup screen is up.
    fn splashing(&self) -> bool {
        self.menu.is_some_and(|menu| menu.screen == Screen::Splash)
    }

    /// Whether the startup screen has done its job: it has been up for a
    /// moment, and the world behind the menu has loaded or is not coming.
    fn splash_over(&self) -> bool {
        let waited = self
            .splash_since
            .is_none_or(|since| since.elapsed() >= SPLASH_AT_LEAST);
        let loaded = self
            .game
            .as_ref()
            .is_none_or(|game| !game.is_panorama() || game.is_on_show());
        waited && loaded
    }

    fn quit_to_title(&mut self) {
        if let Some(game) = self.game.take() {
            game.shutdown();
        }
        if let Some(renderer) = &mut self.renderer {
            renderer.clear_chunks();
        }
        self.hotbar.clear();
        self.catalog.clear();
        self.catalog_blocks.clear();
        self.menu = Some(Menu::new(Screen::Main));
        self.start_title_world();
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

    /// Opens the inventory over a running game.
    fn open_inventory(&mut self) {
        let Some(game) = &mut self.game else {
            return;
        };
        if self.menu.is_none() {
            game.input.clear();
            self.menu = Some(Menu::new(Screen::Inventory));
            self.grab_cursor(false);
        }
    }

    /// Whether `button`, just pressed, closes the inventory that is open.
    fn closes_inventory(&self, button: Button) -> bool {
        let open = self
            .menu
            .is_some_and(|menu| menu.screen == Screen::Inventory);
        let typing = self
            .interface
            .as_ref()
            .is_some_and(Interface::wants_keyboard_input);
        let bound = Button::from_name(self.settings.controls.key(ControlAction::Inventory));
        open && closes_inventory(button, bound, typing)
    }

    /// A block of the inventory goes in a hotbar cell, or two cells swap, as
    /// the player dragged or clicked.
    fn change_hotbar(&mut self, action: MenuAction) {
        let Some(game) = &mut self.game else {
            return;
        };
        match action {
            MenuAction::SetHotbar { slot, block } => {
                if let Some(cell) = self.hotbar.get_mut(slot) {
                    game.set_hotbar(
                        slot,
                        block.and_then(|at| self.catalog_blocks.get(at).copied()),
                    );
                    *cell = block.and_then(|at| self.catalog.get(at).cloned());
                }
            }
            MenuAction::SwapHotbar(a, b) if a < self.hotbar.len() && b < self.hotbar.len() => {
                game.swap_hotbar(a, b);
                self.hotbar.swap(a, b);
            }
            _ => {}
        }
    }

    fn resume(&mut self) {
        if let Some(game) = &mut self.game {
            game.input.clear();
        }
        self.menu = None;
        // A benchmark keeps its view, wherever the mouse goes.
        if self.benchmark.is_none() {
            self.grab_cursor(true);
        }
    }

    fn on_menu_action(&mut self, event_loop: &ActiveEventLoop, action: MenuAction) -> Result<()> {
        match action {
            MenuAction::StartSingleplayer => self.start_game()?,
            MenuAction::Resume => self.resume(),
            MenuAction::QuitToTitle => self.quit_to_title(),
            MenuAction::Exit => event_loop.exit(),
            MenuAction::SettingsClosed => self.save_settings(),
            MenuAction::SetHotbar { .. } | MenuAction::SwapHotbar(..) => {
                self.change_hotbar(action);
            }
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
        let menu_world = (new.menu_world != old.menu_world).then_some(new.menu_world);
        let vsync = self.vsync();
        if new.fps_limit != old.fps_limit
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_vsync(vsync);
        }
        if new.fullscreen != old.fullscreen
            && let Some(window) = &self.window
        {
            window.set_fullscreen(fullscreen(new.fullscreen));
        }
        if new.shadows != old.shadows
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_shadows(shadows(&self.args, new.shadows));
        }
        let clouds = self.clouds();
        if new.lighting != old.lighting
            && let Some(renderer) = &mut self.renderer
        {
            renderer.set_lighting(lighting(new.lighting));
        }
        if let Some(game) = &mut self.game {
            if new.view_distance != old.view_distance {
                game.set_view_distance(new.view_distance);
            }
            if new.fov != old.fov {
                game.set_fov(f32::from(new.fov));
            }
            if new.lod_distance != old.lod_distance {
                game.set_lod_distance(new.lod_distance);
            }
            if new.clouds != old.clouds {
                game.set_clouds(clouds);
            }
            game.set_auto_jump(self.settings.controls.auto_jump);
            game.set_mouse_sensitivity(self.settings.controls.mouse_sensitivity);
            game.set_view_bobbing(new.view_bobbing);
            if self.settings.controls != self.applied.controls {
                game.input.set_bindings(bindings(&self.settings.controls));
            }
        }
        if self.settings.appearance != self.applied.appearance {
            self.refit_interface();
        }
        self.i18n
            .set_language(self.settings.language.unwrap_or(self.system_language));
        self.applied = self.settings.clone();
        match menu_world {
            Some(true) => self.start_title_world(),
            Some(false) => self.stop_title_world(),
            None => {}
        }
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
        if let (Some(game), Some(renderer)) = (&mut self.game, &mut self.renderer) {
            match game.update(dt, self.cursor_grabbed, renderer) {
                Ok(Control::Pause) => self.pause(),
                Ok(Control::Inventory) => self.open_inventory(),
                Ok(Control::Continue) => {}
                // The world behind the menu is not worth ending the program
                // for: the menu goes on with its picture.
                Err(error) if game.is_panorama() => {
                    warn!("the world behind the menu stopped: {error:#}");
                    self.stop_title_world();
                }
                Err(error) => return Err(error),
            }
        }

        if self.loading() && self.game.as_ref().is_some_and(Game::is_ready) {
            self.resume();
        }
        if self.splashing() && self.splash_over() {
            let waited = self.splash_since.map(|since| since.elapsed());
            info!(seconds = ?waited, "the world behind the menu is ready");
            self.menu = Some(Menu::new(Screen::Main));
        }
        let loading = self.loading();
        let world_visible =
            !loading && !self.splashing() && self.game.as_ref().is_some_and(Game::is_on_show);

        let (Some(window), Some(renderer), Some(interface)) =
            (&self.window, &mut self.renderer, &mut self.interface)
        else {
            return Ok(());
        };
        let mut action = None;
        // The HUD is there while a game is, under any menu over it.
        let playing = !loading && self.game.as_ref().is_some_and(|game| !game.is_panorama());
        let painted = if self.menu.is_some() || playing {
            let images = interface.images.clone();
            let context = MenuContext {
                i18n: &self.i18n,
                images: &images,
                inventory: InventoryContext {
                    catalog: &self.catalog,
                    hotbar: &self.hotbar,
                    selected: self.game.as_ref().map_or(0, Game::selected),
                },
                max_scale: Interface::max_scale(window),
                world_visible,
                version: env!("CARGO_PKG_VERSION"),
                system_language: self.system_language,
            };
            let hud = HudContext {
                i18n: &self.i18n,
                images: &images,
                slots: &self.hotbar,
                selected: self.game.as_ref().map_or(0, Game::selected),
            };
            let (menu, settings, hud_state) = (&mut self.menu, &mut self.settings, &mut self.hud);
            // With no menu the game owns the cursor, which the interface
            // must leave alone.
            let cursor = menu.is_some();
            Some(interface.run(window, cursor, |ui| {
                if playing {
                    hud_state.show(ui, hud);
                }
                if let Some(menu) = menu
                    && let Some(clicked) = menu.show(ui, context, settings)
                {
                    action = Some(clicked);
                }
            }))
        } else {
            interface.skip_frame(window);
            None
        };
        let ui = painted.as_ref().map(|painted| painted.frame());
        let scene = self.game.as_ref().filter(|_| world_visible).map(|game| {
            let mut scene = game.scene();
            // Nothing to aim with while a menu is open.
            scene.crosshair &= self.menu.is_none();
            scene
        });
        let backdrop = match &scene {
            Some(scene) => Backdrop::World(scene),
            None => {
                let color = ruda_ui::BACKGROUND;
                Backdrop::Color([color.r(), color.g(), color.b()])
            }
        };
        let presented = renderer.render(backdrop, ui.as_ref(), || window.pre_present_notify())?;
        if presented && !loading {
            self.frames += 1;
            self.first_frame_at.get_or_insert(now);
        }
        if presented {
            self.title.frames += 1;
            #[cfg(feature = "tracy")]
            tracing_tracy::client::frame_mark();
            let busy = now.elapsed().saturating_sub(renderer.surface_wait());
            let gpu = renderer.take_gpu_times();
            if let (Some(benchmark), Some(game)) = (&mut self.benchmark, &self.game)
                && let Some(report) = benchmark.frame(now, busy, game.is_loaded(), gpu)
            {
                let stats = renderer.stats();
                info!("benchmark: {report}; {stats}");
                println!("{report}\n{stats}");
                if let Some(path) = &self.args.screenshot {
                    let (width, height, pixels) = renderer.capture(backdrop, ui.as_ref())?;
                    save_png(path, width, height, &pixels)?;
                }
                event_loop.exit();
            }
        }

        let elapsed = now - self.title.since;
        if elapsed >= Duration::from_millis(500) {
            let fps = f64::from(self.title.frames) / elapsed.as_secs_f64();
            let status = self.game.as_ref().map(Game::status).unwrap_or_default();
            window.set_title(&if status.is_empty() {
                format!("Ruda — {fps:.0} FPS")
            } else {
                format!("Ruda — {fps:.0} FPS · {status}")
            });
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
        } else if !presented {
            // The window can't show frames right now, for example while the
            // screen is locked: try again a little later instead of spinning.
            self.repaint_at = Some(now + Duration::from_millis(16));
        } else if repaint_after.is_zero() {
            // With a limit, the next frame waits for its turn.
            match self.frame_interval() {
                Some(interval) if now + interval > Instant::now() => {
                    self.repaint_at = Some(now + interval);
                }
                _ => window.request_redraw(),
            }
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

    /// Gives the button in `event` to the action the player is picking one
    /// for, if they are. The left mouse button is left out: it is what clicks
    /// the menu, and Escape gives up.
    fn capture_button(&mut self, event: &WindowEvent) -> bool {
        let Some(menu) = self.menu.as_mut().filter(|menu| menu.is_listening()) else {
            return false;
        };
        let button = match event {
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } if *code != KeyCode::Escape => Button::Key(*code),
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } if *button != MouseButton::Left => Button::Mouse(*button),
            _ => return false,
        };
        // The number keys pick hotbar cells and Escape pauses: they stay as
        // they are, and the player goes on picking.
        if is_reserved(button) {
            return true;
        }
        let Some(name) = button.name() else {
            return false;
        };
        menu.offer_button(&mut self.settings, &name);
        self.request_redraw();
        true
    }

    /// Sizes the menus again, for a new scale or a window that changed.
    fn refit_interface(&mut self) {
        if let (Some(interface), Some(window)) = (&mut self.interface, &self.window) {
            interface.set_scale(window, self.settings.appearance.scale);
        }
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
        // A button pressed while the player picks one for an action is for
        // that, and for nothing else.
        if self.capture_button(&event) {
            return;
        }
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
            // Menus fit the window, whose size in points depends on the scale.
            WindowEvent::ScaleFactorChanged { .. } => self.refit_interface(),
            WindowEvent::Resized(size) => {
                self.zero_sized = size.width == 0 || size.height == 0;
                if let Some(renderer) = &mut self.renderer {
                    renderer.resize(size.width, size.height);
                }
                self.refit_interface();
                self.request_redraw();
            }
            WindowEvent::Occluded(occluded) => {
                self.occluded = occluded;
                if !occluded {
                    self.request_redraw();
                }
            }
            // The key that opens the inventory closes it too.
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        physical_key: PhysicalKey::Code(code),
                        state: ElementState::Pressed,
                        repeat: false,
                        ..
                    },
                ..
            } if self.closes_inventory(Button::Key(code)) => {
                self.resume();
                self.request_redraw();
            }
            // Or the mouse button it is bound to, if it is one.
            WindowEvent::MouseInput {
                state: ElementState::Pressed,
                button,
                ..
            } if self.closes_inventory(Button::Mouse(button)) => {
                self.resume();
                self.request_redraw();
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
            } if self.menu.is_none() && !self.cursor_grabbed && self.benchmark.is_none() => {
                self.grab_cursor(true);
                if let Some(game) = &mut self.game {
                    game.input.clear();
                }
            }
            // Automated runs keep going in the background.
            WindowEvent::Focused(false)
                if self.benchmark.is_none() && self.args.exit_after_frames.is_none() =>
            {
                self.pause()
            }
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

/// The bindings the settings ask for: the usual ones, and the buttons the
/// player picked for actions they can pick for.
fn bindings(controls: &Controls) -> Bindings {
    let mut bindings = Bindings::default();
    for (action, name) in controls.keys() {
        // A name this version doesn't know, or a button that is not for
        // picking, is left to the usual one.
        if let Some(button) = Button::from_name(name)
            && !is_reserved(button)
        {
            bindings.rebind(input_action(action), button);
        }
    }
    bindings
}

/// Whether `button` is one the player cannot pick for an action: it picks a
/// hotbar cell or pauses.
fn is_reserved(button: Button) -> bool {
    static USUAL: std::sync::OnceLock<Bindings> = std::sync::OnceLock::new();
    matches!(
        USUAL.get_or_init(Bindings::default).action(button),
        Some(Action::Hotbar(_) | Action::Pause)
    )
}

/// Whether pressing `button` closes an inventory that is open: it is the one
/// the inventory is bound to, and for a key, nothing is being typed in a
/// text field (a mouse button types nothing).
fn closes_inventory(button: Button, bound: Option<Button>, typing: bool) -> bool {
    bound == Some(button) && !(typing && matches!(button, Button::Key(_)))
}

fn input_action(action: ControlAction) -> Action {
    match action {
        ControlAction::MoveForward => Action::MoveForward,
        ControlAction::MoveBack => Action::MoveBack,
        ControlAction::MoveLeft => Action::MoveLeft,
        ControlAction::MoveRight => Action::MoveRight,
        ControlAction::Jump => Action::Jump,
        ControlAction::Sneak => Action::Sneak,
        ControlAction::Sprint => Action::Sprint,
        ControlAction::Break => Action::Break,
        ControlAction::Place => Action::Place,
        ControlAction::Inventory => Action::Inventory,
    }
}

/// Whether there are clouds: as `saved` in the settings, unless the command
/// line says otherwise.
fn clouds(args: &Args, saved: bool) -> bool {
    saved && !args.no_clouds
}

/// `--benchmark --headless`: the game without a window, each frame drawn
/// off-screen and waited for, then the results printed.
fn run_headless(args: &Args) -> Result<()> {
    let settings = load_settings(settings::path().as_deref(), args);
    let graphics = &settings.graphics;
    let (width, height) = args.size;
    let mut renderer = pollster::block_on(Renderer::headless(width, height))?;
    info!(adapter = %renderer.adapter_summary(), width, height, "drawing off-screen");
    renderer.set_shadows(shadows(args, graphics.shadows));
    let clouds = clouds(args, graphics.clouds);
    renderer.set_lighting(lighting(graphics.lighting));
    let config = GameConfig {
        seed: args.seed.unwrap_or_else(random_seed),
        view_distance: args.view_distance.unwrap_or(graphics.view_distance),
        fov: f32::from(graphics.fov),
        camera: args.camera,
        time: args.time,
        lod_distance: args.lod_distance.unwrap_or(graphics.lod_distance),
        clouds,
        auto_jump: false,
        mouse_sensitivity: settings.controls.mouse_sensitivity,
        view_bobbing: graphics.view_bobbing,
        panorama: false,
    };
    let mut game = Game::start(config, &mut renderer)?;
    let seconds = args.benchmark.context("--headless needs --benchmark")?;
    let mut benchmark = Benchmark::new(Duration::from_secs_f64(seconds));
    let mut last = Instant::now();
    loop {
        let now = Instant::now();
        game.update((now - last).as_secs_f64(), false, &mut renderer)?;
        last = now;
        let scene = game.scene();
        renderer.render_offscreen(Backdrop::World(&scene))?;
        let gpu = renderer.take_gpu_times();
        if let Some(report) = benchmark.frame(now, now.elapsed(), game.is_loaded(), gpu) {
            println!("{report}\n{}", renderer.stats());
            if let Some(path) = &args.screenshot {
                let (width, height, pixels) = renderer.capture(Backdrop::World(&scene), None)?;
                save_png(path, width, height, &pixels)?;
            }
            break;
        }
    }
    game.shutdown();
    Ok(())
}

/// How the renderer casts shadows: as `saved` in the settings, unless the
/// command line says otherwise.
fn shadows(args: &Args, saved: Shadows) -> ruda_render::Shadows {
    match args.shadows.map_or(saved, Shadows::from) {
        Shadows::Off => ruda_render::Shadows::Off,
        Shadows::Standard => ruda_render::Shadows::Map,
        Shadows::Rays => ruda_render::Shadows::Rays,
    }
}

/// How the renderer lights the world.
fn lighting(lighting: Lighting) -> ruda_render::Lighting {
    match lighting {
        Lighting::Classic => ruda_render::Lighting::Classic,
        Lighting::Atmospheric => ruda_render::Lighting::Atmospheric,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_settings_start_from_the_usual_bindings() {
        let usual = Bindings::default();
        for action in ControlAction::ALL {
            let button = Button::from_name(action.default_key());
            assert!(button.is_some(), "{action:?}");
            assert_eq!(button, usual.button(input_action(action)), "{action:?}");
        }
    }

    #[test]
    fn the_button_that_opens_the_inventory_closes_it_unless_typing() {
        let key = Button::from_name("KeyE").unwrap();
        let mouse = Button::from_name("mouse:Middle").unwrap();
        let other = Button::from_name("KeyF").unwrap();
        assert!(closes_inventory(key, Some(key), false));
        assert!(!closes_inventory(key, Some(key), true));
        assert!(!closes_inventory(other, Some(key), false));
        // A mouse button works the same, and a text field does not hold it.
        assert!(closes_inventory(mouse, Some(mouse), false));
        assert!(closes_inventory(mouse, Some(mouse), true));
        assert!(!closes_inventory(key, None, false));
    }

    #[test]
    fn the_hotbar_and_the_pause_cannot_be_taken() {
        let mut controls = Controls::default();
        controls.set_key(ControlAction::Jump, "Digit1");
        // Even a file that says so does not take the key from the hotbar.
        let picked = bindings(&controls);
        assert_eq!(
            picked.action(Button::from_name("Digit1").unwrap()),
            Some(Action::Hotbar(0))
        );
        assert_eq!(
            picked.button(Action::Jump),
            Bindings::default().button(Action::Jump)
        );
        assert!(is_reserved(Button::from_name("Escape").unwrap()));
        assert!(!is_reserved(Button::from_name("KeyF").unwrap()));
    }

    #[test]
    fn picked_buttons_replace_the_usual_ones() {
        let mut controls = Controls::default();
        controls.set_key(ControlAction::Jump, "KeyF");
        let picked = bindings(&controls);
        assert_eq!(picked.button(Action::Jump), Button::from_name("KeyF"));
        // The rest keep theirs, including what the settings can't change.
        assert_eq!(
            picked.button(Action::Sprint),
            Bindings::default().button(Action::Sprint)
        );
        assert_eq!(
            picked.button(Action::Pause),
            Bindings::default().button(Action::Pause)
        );
    }

    #[test]
    fn images_decode() {
        super::window_icon().unwrap();
        for png in [
            super::interface::LOGO_PNG,
            super::interface::BACKGROUND_PNG,
            super::interface::COBBLESTONE_PNG,
        ] {
            super::decode_png(png).unwrap();
        }
    }
}
