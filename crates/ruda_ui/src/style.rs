use std::sync::Arc;

use egui::{
    Color32, CornerRadius, FontData, FontDefinitions, FontFamily, FontId, FontTweak, Stroke,
    TextStyle, Vec2,
};

const PIXEL_FONT: &[u8] = include_bytes!("../../../assets/fonts/Tiny5-Regular.ttf");

/// Behind menus that have no world to show, matching the dark logo.
pub const BACKGROUND: Color32 = Color32::from_rgb(18, 20, 25);
const PANEL: Color32 = Color32::from_rgb(28, 31, 38);
const BUTTON: Color32 = Color32::from_rgb(44, 48, 58);
const BUTTON_HOVER: Color32 = Color32::from_rgb(60, 65, 78);
/// The thin edge of what is not stone: tooltips, hollows.
const RIM: Color32 = Color32::from_rgb(52, 56, 66);
/// The outline of stone pieces.
pub const OUTLINE: Color32 = Color32::from_rgb(9, 10, 13);
/// The orange of the ore in the logo.
pub const ACCENT: Color32 = Color32::from_rgb(255, 138, 42);
pub const ACCENT_LIGHT: Color32 = Color32::from_rgb(255, 196, 134);
pub const TEXT: Color32 = Color32::from_rgb(236, 238, 242);

/// Every piece of text is set in the one pixel font.
pub fn pixel_family() -> FontFamily {
    FontFamily::Name("pixel".into())
}

/// A font of the pixel family. Its pixels are an eighth of the size, so
/// sizes that are multiples of 8 keep them sharp; 4 does at twice the
/// scale.
pub fn pixel(size: f32) -> FontId {
    FontId::new(size, pixel_family())
}

/// The game's font, sizes and colours.
pub fn apply(ctx: &egui::Context) {
    let mut fonts = FontDefinitions::default();
    // Snapping glyphs to whole pixels is what keeps the pixel font crisp.
    fonts.font_data.insert(
        "pixel".into(),
        Arc::new(FontData::from_static(PIXEL_FONT).tweak(FontTweak {
            hinting: Some(false),
            subpixel_binning: Some(false),
            ..Default::default()
        })),
    );
    // Whatever asks for the usual fonts, such as egui's own widgets, gets it
    // too.
    for family in [
        FontFamily::Proportional,
        FontFamily::Monospace,
        pixel_family(),
    ] {
        fonts.families.insert(family, vec!["pixel".into()]);
    }
    ctx.set_fonts(fonts);

    ctx.all_styles_mut(|style| {
        style.text_styles = [
            (TextStyle::Heading, pixel(40.0)),
            (TextStyle::Button, pixel(24.0)),
            (TextStyle::Body, pixel(24.0)),
            (TextStyle::Monospace, pixel(24.0)),
            (TextStyle::Small, pixel(16.0)),
        ]
        .into_iter()
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
        // Square, like everything else; what is left of egui's own widgets
        // (the search field, scroll bars, tooltips) matches the stone.
        let radius = CornerRadius::ZERO;
        visuals.window_corner_radius = radius;
        visuals.menu_corner_radius = radius;
        visuals.window_stroke = Stroke::new(2.0, RIM);
        visuals.window_shadow = egui::Shadow::NONE;
        visuals.popup_shadow = egui::Shadow::NONE;
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
