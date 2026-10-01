use std::sync::Arc;

use egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, Stroke, TextStyle, Vec2,
};

const FONT: &[u8] = include_bytes!("../../../assets/fonts/PixelifySans.ttf");

/// Background of menus, matching the dark logo so it blends in.
pub const BACKGROUND: Color32 = Color32::from_rgb(18, 20, 25);
const PANEL: Color32 = Color32::from_rgb(28, 31, 38);
const BUTTON: Color32 = Color32::from_rgb(44, 48, 58);
const BUTTON_HOVER: Color32 = Color32::from_rgb(60, 65, 78);
/// The orange of the ore in the logo.
const ACCENT: Color32 = Color32::from_rgb(232, 128, 48);
const TEXT: Color32 = Color32::from_rgb(236, 238, 242);

/// The game's font, sizes and colours.
pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    fonts
        .font_data
        .insert("pixelify".into(), Arc::new(FontData::from_static(FONT)));
    for family in [FontFamily::Proportional, FontFamily::Monospace] {
        fonts.families.insert(family, vec!["pixelify".into()]);
    }
    ctx.set_fonts(fonts);

    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, 36.0),
            (TextStyle::Button, 24.0),
            (TextStyle::Body, 20.0),
            (TextStyle::Monospace, 18.0),
            (TextStyle::Small, 15.0),
        ]
        .into_iter()
        .map(|(style, size)| (style, FontId::proportional(size)))
        .collect();
        style.spacing.item_spacing = Vec2::new(12.0, 12.0);
        style.spacing.button_padding = Vec2::new(18.0, 8.0);
        style.spacing.interact_size.y = 32.0;
        style.spacing.slider_width = 220.0;

        let visuals = &mut style.visuals;
        *visuals = egui::Visuals::dark();
        visuals.panel_fill = BACKGROUND;
        visuals.window_fill = PANEL;
        visuals.override_text_color = None;
        visuals.selection.bg_fill = ACCENT;
        visuals.selection.stroke = Stroke::new(1.0, TEXT);
        let radius = CornerRadius::same(4);
        for (widget, fill) in [
            (&mut visuals.widgets.inactive, BUTTON),
            (&mut visuals.widgets.hovered, BUTTON_HOVER),
            (&mut visuals.widgets.active, ACCENT),
            (&mut visuals.widgets.open, BUTTON_HOVER),
        ] {
            widget.bg_fill = fill;
            widget.weak_bg_fill = fill;
            widget.corner_radius = radius;
            widget.fg_stroke = Stroke::new(1.0, TEXT);
            widget.bg_stroke = Stroke::NONE;
        }
        visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, ACCENT);
        visuals.widgets.noninteractive.fg_stroke = Stroke::new(1.0, TEXT);
        visuals.widgets.noninteractive.corner_radius = radius;
        visuals.widgets.noninteractive.bg_stroke = Stroke::NONE;
    });
}
