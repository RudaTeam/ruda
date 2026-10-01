//! The egui side of the window: feeds it events and turns what it draws into
//! something the renderer can paint over the game.

use std::time::Duration;

use anyhow::Result;
use egui::{ClippedPrimitive, TextureHandle, TexturesDelta, ViewportId};
use ruda_render::UiFrame;
use winit::event::WindowEvent;
use winit::window::Window;

use crate::decode_png;

pub const LOGO_PNG: &[u8] = include_bytes!("../../../assets/branding/logo-dark.png");

pub struct Interface {
    ctx: egui::Context,
    state: egui_winit::State,
    pub logo: Option<TextureHandle>,
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
        let logo = load_logo(&ctx)
            .inspect_err(|error| tracing::warn!("no logo: {error:#}"))
            .ok();
        Self { ctx, state, logo }
    }

    /// Returns whether the interface needs drawing again.
    pub fn window_event(&mut self, window: &Window, event: &WindowEvent) -> bool {
        self.state.on_window_event(window, event).repaint
    }

    /// Lays out and paints a frame.
    pub fn run(&mut self, window: &Window, ui: impl FnMut(&mut egui::Ui)) -> Painted {
        let input = self.state.take_egui_input(window);
        let output = self.ctx.run_ui(input, ui);
        self.state
            .handle_platform_output(window, output.platform_output);
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

fn load_logo(ctx: &egui::Context) -> Result<TextureHandle> {
    let (width, height, rgba) = decode_png(LOGO_PNG)?;
    let image = egui::ColorImage::from_rgba_unmultiplied([width as usize, height as usize], &rgba);
    Ok(ctx.load_texture("logo", image, egui::TextureOptions::LINEAR))
}
