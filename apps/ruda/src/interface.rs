//! The egui side of the window: feeds it events and turns what it draws into
//! something the renderer can paint over the game.

use std::time::Duration;

use egui::{ClippedPrimitive, TextureHandle, TextureOptions, TexturesDelta, ViewportId};
use ruda_core::BlockId;
use ruda_render::UiFrame;
use ruda_ui::{HotbarSlot, Images, PICTURE_TEXTURE, TILING_TEXTURE, UI_SCALE_STEP, UI_SCALES};
use winit::event::WindowEvent;
use winit::window::Window;

use crate::decode_png;
use crate::game::Game;
use crate::icons::block_icon;

pub const LOGO_PNG: &[u8] = include_bytes!("../../../assets/branding/menu-logo.png");
pub const GRASS_PNG: &[u8] = include_bytes!("../../../content/base/textures/grass_side.png");
pub const DIRT_PNG: &[u8] = include_bytes!("../../../content/base/textures/dirt.png");
pub const STONE_PNG: &[u8] = include_bytes!("../../../content/base/textures/stone.png");
pub const COBBLESTONE_PNG: &[u8] = include_bytes!("../../../content/base/textures/cobblestone.png");

/// The area, in points, that menus are laid out to fit at least.
const MENU_AREA: egui::Vec2 = egui::vec2(960.0, 600.0);

/// How many times larger than normal menus could be drawn and still fit the
/// window.
fn fit(window: &Window) -> f32 {
    let size = window.inner_size();
    let native = window.scale_factor() as f32;
    (size.width as f32 / native / MENU_AREA.x).min(size.height as f32 / native / MENU_AREA.y)
}

pub struct Interface {
    ctx: egui::Context,
    state: egui_winit::State,
    pub images: Images,
}

/// One frame of the interface, ready to draw.
pub struct Painted {
    primitives: Vec<ClippedPrimitive>,
    textures: TexturesDelta,
    pixels_per_point: f32,
    /// How soon egui wants to draw again, for animations and tooltips.
    pub repaint_after: Duration,
}

impl Interface {
    pub fn new(window: &Window, max_texture_side: usize) -> Self {
        let ctx = egui::Context::default();
        ruda_ui::apply_style(&ctx);
        let state = egui_winit::State::new(
            ctx.clone(),
            ViewportId::ROOT,
            window,
            Some(window.scale_factor() as f32),
            None,
            Some(max_texture_side),
        );
        let images = Images {
            logo: load_image(&ctx, "logo", LOGO_PNG, PICTURE_TEXTURE),
            cobblestone: load_image(&ctx, "cobblestone", COBBLESTONE_PNG, TILING_TEXTURE),
            sun: load_image(&ctx, "sun", ruda_base::SUN, TextureOptions::NEAREST),
            grass: load_image(&ctx, "grass", GRASS_PNG, TILING_TEXTURE),
            dirt: load_image(&ctx, "dirt", DIRT_PNG, TILING_TEXTURE),
            stone: load_image(&ctx, "stone", STONE_PNG, TILING_TEXTURE),
        };
        Self { ctx, state, images }
    }

    /// How large menus are drawn, in percent, but no larger than lets them
    /// fit the window: a panel that sticks out of a small window can't be
    /// used. Call it again when the window changes size.
    pub fn set_scale(&mut self, window: &Window, percent: u8) {
        // Never below half, however small the window.
        let zoom = (f32::from(percent) / 100.0).min(fit(window).max(0.5));
        self.ctx.set_zoom_factor(zoom);
    }

    /// The largest size menus can be offered at for this window, in percent,
    /// a whole step of the sizes there are.
    pub fn max_scale(window: &Window) -> u8 {
        let step = u32::from(UI_SCALE_STEP);
        let steps = (fit(window) * 100.0 / step as f32).floor() as u32;
        let (smallest, largest) = (*UI_SCALES.start(), *UI_SCALES.end());
        (steps * step).clamp(u32::from(smallest), u32::from(largest)) as u8
    }

    /// Returns whether the interface needs drawing again.
    pub fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.state.on_window_event(window, event).repaint
    }

    /// Pictures of `blocks`, for the HUD and the inventory.
    pub fn block_slots(&self, game: &Game, blocks: &[BlockId]) -> Vec<HotbarSlot> {
        let content = game.content();
        blocks
            .iter()
            .map(|&block| {
                let id = content.blocks().get(block).id.as_str().to_owned();
                let icon = block_icon(content, block).map(|image| {
                    self.ctx
                        .load_texture(format!("icon {id}"), image, PICTURE_TEXTURE)
                });
                HotbarSlot { block: id, icon }
            })
            .collect()
    }

    /// Whether the interface has a text field to type in: keys then belong
    /// to it.
    pub fn wants_keyboard_input(&self) -> bool {
        self.ctx.egui_wants_keyboard_input()
    }

    /// Lays out and paints a frame. Unless `platform_output` is false, what
    /// the interface asks of the window, such as the shape of the cursor,
    /// is done: not while playing, when the game has hidden the cursor.
    pub fn run(
        &mut self,
        window: &Window,
        platform_output: bool,
        ui: impl FnMut(&mut egui::Ui),
    ) -> Painted {
        let input = self.state.take_egui_input(window);
        let output = self.ctx.run_ui(input, ui);
        if platform_output {
            self.state
                .handle_platform_output(window, output.platform_output);
        }
        let repaint_after = output
            .viewport_output
            .get(&ViewportId::ROOT)
            .map_or(Duration::MAX, |viewport| viewport.repaint_delay);
        Painted {
            primitives: self.ctx.tessellate(output.shapes, output.pixels_per_point),
            textures: output.textures_delta,
            pixels_per_point: output.pixels_per_point,
            repaint_after,
        }
    }

    /// Drops input gathered while nothing was shown, so it neither piles up
    /// nor reaches the next menu.
    pub fn skip_frame(&mut self, window: &Window) {
        self.state.take_egui_input(window);
    }
}

impl Painted {
    pub fn frame(&self) -> UiFrame<'_> {
        UiFrame {
            primitives: &self.primitives,
            textures: &self.textures,
            pixels_per_point: self.pixels_per_point,
        }
    }
}

impl Drop for Painted {
    fn drop(&mut self) {
        // Every frame goes to the renderer, which applies the texture changes
        // even when the frame doesn't reach the screen.
        self.textures.clear();
    }
}

/// Menus manage without an image that fails to load, so this only warns.
fn load_image(
    ctx: &egui::Context,
    name: &str,
    png: &[u8],
    options: TextureOptions,
) -> Option<TextureHandle> {
    let image = decode_png(png).map(|(width, height, rgba)| {
        egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba)
    });
    match image {
        Ok(image) => Some(ctx.load_texture(name, image, options)),
        Err(error) => {
            tracing::warn!("no {name} image: {error:#}");
            None
        }
    }
}
