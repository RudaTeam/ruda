//! Widgets for setting things, in the same stone: a button that steps
//! through values, a slider and a tab.

use std::ops::RangeInclusive;

use egui::{
    Color32, EventFilter, Key, Pos2, Rect, Response, Sense, Shape, Stroke, Ui, WidgetInfo,
    WidgetType, emath::GuiRounding, pos2, vec2,
};

use crate::style::{ACCENT, ACCENT_LIGHT, TEXT, pixel};
use crate::widgets::{EDGE, Images, TEXEL, bevel, glow, pixel_label, stone_piece, sunken};

/// Every setting is this big, whatever it is set by.
pub(crate) const CONTROL_SIZE: egui::Vec2 = egui::Vec2::new(300.0, 40.0);
pub(crate) const TAB_SIZE: egui::Vec2 = egui::Vec2::new(230.0, 48.0);
/// How wide a slider's handle is.
const HANDLE_WIDTH: f32 = 24.0;
const ACCENT_DARK: Color32 = Color32::from_rgb(201, 105, 31);

/// Lets the focused widget have the left and right arrow keys, which
/// otherwise move the focus to the next widget over.
fn keep_horizontal_arrows(ui: &Ui, response: &Response) {
    if response.has_focus() {
        ui.memory_mut(|memory| {
            memory.set_focus_lock_filter(
                response.id,
                EventFilter {
                    horizontal_arrows: true,
                    ..Default::default()
                },
            );
        });
    }
}

/// -1 or 1 for the left or right arrow pressed while `response` has focus.
fn arrow_step(ui: &Ui, response: &Response) -> i32 {
    if !response.has_focus() {
        return 0;
    }
    ui.input(|input| {
        i32::from(input.key_pressed(Key::ArrowRight)) - i32::from(input.key_pressed(Key::ArrowLeft))
    })
}

/// A stone button showing a value, which steps to the next one when clicked
/// and to the previous one on a right click or the left arrow key. Returns
/// -1 or 1 for the step taken, 0 for none, and the button's response.
pub(crate) fn cycle(
    ui: &mut Ui,
    images: &Images,
    text: &str,
    color: Option<Color32>,
) -> (i32, Response) {
    let (rect, response) = ui.allocate_exact_size(CONTROL_SIZE, Sense::click());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, text));
    keep_horizontal_arrows(ui, &response);
    let step = if response.clicked() {
        1
    } else if response.secondary_clicked() {
        -1
    } else {
        arrow_step(ui, &response)
    };
    if ui.is_rect_visible(rect) {
        let rect = rect.round_to_pixels(ui.pixels_per_point());
        let glow = glow(ui, &response, true);
        let pressed = response.is_pointer_button_down_on();
        stone_piece(ui.painter(), images, rect, glow, pressed, true);
        let base = color.unwrap_or(TEXT);
        let color = base.lerp_to_gamma(ACCENT_LIGHT, glow * 0.7);
        pixel_label(ui.painter(), rect.center(), text, color, true);
        // Little arrows on the sides say it can be stepped either way.
        if glow > 0.0 {
            let arrow = ACCENT_LIGHT.gamma_multiply(glow);
            for (x, towards) in [(rect.min.x + 20.0, -1.0), (rect.max.x - 20.0, 1.0)] {
                triangle(ui, pos2(x, rect.center().y), towards, arrow);
            }
        }
    }
    (step, response)
}

/// A stone button for a button of the keyboard or mouse picked for an action;
/// lit up while the player is picking.
pub(crate) fn key_button(ui: &mut Ui, images: &Images, text: &str, picking: bool) -> Response {
    let (rect, response) = ui.allocate_exact_size(CONTROL_SIZE, Sense::click());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, text));
    if ui.is_rect_visible(rect) {
        let rect = rect.round_to_pixels(ui.pixels_per_point());
        let glow = glow(ui, &response, true).max(if picking { 1.0 } else { 0.0 });
        let pressed = response.is_pointer_button_down_on();
        stone_piece(ui.painter(), images, rect, glow, pressed, true);
        let color = TEXT.lerp_to_gamma(ACCENT_LIGHT, glow);
        pixel_label(ui.painter(), rect.center(), text, color, true);
    }
    response
}

/// A small triangle pointing left (`towards` -1) or right (1).
fn triangle(ui: &Ui, center: Pos2, towards: f32, color: Color32) {
    let points = vec![
        center + vec2(-2.0 * TEXEL * towards, -2.0 * TEXEL * 1.5),
        center + vec2(-2.0 * TEXEL * towards, 2.0 * TEXEL * 1.5),
        center + vec2(2.0 * TEXEL * towards, 0.0),
    ];
    ui.painter()
        .add(Shape::convex_polygon(points, color, Stroke::NONE));
}

/// A slider through `range` in whole `step`s, with `text` written over it.
/// Dragging, clicking and the arrow keys move it.
pub(crate) fn slider(
    ui: &mut Ui,
    images: &Images,
    value: &mut f32,
    range: RangeInclusive<f32>,
    step: f32,
    text: &str,
) -> Response {
    let (rect, mut response) = ui.allocate_exact_size(CONTROL_SIZE, Sense::click_and_drag());
    response.widget_info(|| WidgetInfo::slider(true, f64::from(*value), text));
    keep_horizontal_arrows(ui, &response);
    let (low, high) = (*range.start(), *range.end());
    let rect = rect.round_to_pixels(ui.pixels_per_point());
    let travel = rect.width() - HANDLE_WIDTH;
    let snap = |value: f32| (((value - low) / step).round() * step + low).clamp(low, high);

    let before = *value;
    if response.is_pointer_button_down_on()
        && let Some(pointer) = response.interact_pointer_pos()
    {
        let along = ((pointer.x - rect.min.x - HANDLE_WIDTH / 2.0) / travel).clamp(0.0, 1.0);
        *value = snap(low + along * (high - low));
    }
    let arrows = arrow_step(ui, &response);
    if arrows != 0 {
        *value = snap(*value + arrows as f32 * step);
    }
    if *value != before {
        response.mark_changed();
    }

    if ui.is_rect_visible(rect) {
        let glow = glow(ui, &response, true);
        let painter = ui.painter();
        let along = (*value - low) / (high - low);
        let handle = Rect::from_min_size(
            pos2(rect.min.x + along * travel, rect.min.y),
            vec2(HANDLE_WIDTH, rect.height()),
        )
        .round_to_pixels(ui.pixels_per_point());

        // The hollow the handle slides in, filled with ore up to the handle.
        painter.rect_stroke(
            rect,
            0.0,
            Stroke::new(EDGE, Color32::from_rgb(9, 10, 13)),
            egui::StrokeKind::Outside,
        );
        sunken(painter, rect);
        let inner = rect.shrink(EDGE);
        let filled = Rect::from_min_max(inner.min, pos2(handle.center().x, inner.max.y));
        let painter_clipped = painter.with_clip_rect(inner);
        // Darker ore under the text, so it can be read, with a bright top.
        painter_clipped.rect_filled(filled, 0.0, ACCENT_DARK);
        painter_clipped.rect_filled(
            Rect::from_min_size(filled.min, vec2(filled.width(), EDGE)),
            0.0,
            ACCENT,
        );
        stone_piece(painter, images, handle, glow, false, true);
        // A dark plate under the value, so the handle and the ore don't
        // get in the way of reading it.
        let value = painter.layout_no_wrap(text.to_owned(), pixel(24.0), TEXT);
        let plate = Rect::from_center_size(rect.center(), value.size() + vec2(16.0, 4.0));
        painter.rect_filled(plate, 0.0, Color32::from_black_alpha(130));
        pixel_label(painter, rect.center(), text, TEXT, true);
    }
    response
}

/// A tab in a column of them; the chosen one has a bar of ore beside it.
pub(crate) fn tab(ui: &mut Ui, text: &str, selected: bool, enabled: bool) -> Response {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(TAB_SIZE, sense);
    response
        .widget_info(|| WidgetInfo::selected(WidgetType::SelectableLabel, enabled, selected, text));
    if ui.is_rect_visible(rect) {
        let rect = rect.round_to_pixels(ui.pixels_per_point());
        let glow = glow(ui, &response, enabled);
        let painter = ui.painter();
        sunken(painter, rect);
        if selected {
            painter.rect_filled(rect, 0.0, ACCENT.gamma_multiply(0.18));
            let bar = Rect::from_min_size(rect.min, vec2(EDGE, rect.height()));
            painter.rect_filled(bar, 0.0, ACCENT);
        } else if glow > 0.0 {
            painter.rect_filled(rect, 0.0, Color32::from_white_alpha((22.0 * glow) as u8));
            bevel(
                painter,
                rect,
                Color32::from_white_alpha((40.0 * glow) as u8),
                Color32::from_black_alpha(0),
            );
        }
        let color = if !enabled {
            TEXT.gamma_multiply(0.4)
        } else if selected {
            ACCENT_LIGHT
        } else {
            TEXT.lerp_to_gamma(ACCENT_LIGHT, glow)
        };
        let label = painter.layout_no_wrap(text.to_owned(), pixel(24.0), color);
        let at = pos2(
            rect.min.x + EDGE + 2.0 * TEXEL * 4.0,
            rect.center().y - label.size().y / 2.0,
        )
        .round_to_pixels(ui.pixels_per_point());
        if enabled {
            let dark = painter.layout_no_wrap(
                text.to_owned(),
                pixel(24.0),
                Color32::from_black_alpha(166),
            );
            painter.galley(at + egui::Vec2::splat(TEXEL), dark, Color32::PLACEHOLDER);
        }
        painter.galley(at, label, Color32::PLACEHOLDER);
    }
    response
}
