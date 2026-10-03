//! What is on the screen while playing: the hotbar and the name of the
//! block in hand.

use egui::epaint::Shadow;
use egui::{
    Align2, Color32, Rect, Stroke, StrokeKind, TextureHandle, Ui, emath::GuiRounding, pos2, vec2,
};

use crate::I18n;
use crate::style::{ACCENT, ACCENT_LIGHT, pixel};
use crate::widgets::{EDGE, Images, TEXEL, panel, pixel_text, sunken};

/// A cell of the hotbar is this big, with this much between.
pub(crate) const SLOT: f32 = 56.0;
pub(crate) const GAP: f32 = 4.0;
/// A block's picture in its cell.
pub(crate) const ICON: f32 = 44.0;
/// Between the hotbar and the bottom of the screen.
const BOTTOM: f32 = 20.0;
/// Seconds the name of the block in hand stays after it changes, and
/// takes to fade.
const NAME_SHOWN: f64 = 2.0;
const NAME_FADE: f64 = 0.6;

/// A cell of the hotbar.
#[derive(Clone)]
pub struct HotbarSlot {
    /// The block's id, such as `base:stone`, for its translated name.
    pub block: String,
    pub icon: Option<TextureHandle>,
}

impl std::fmt::Debug for HotbarSlot {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HotbarSlot")
            .field("block", &self.block)
            .finish_non_exhaustive()
    }
}

/// What the HUD shows but does not own.
#[derive(Clone, Copy)]
pub struct HudContext<'a> {
    pub i18n: &'a I18n,
    pub images: &'a Images,
    /// What each cell holds; `None` for an empty one.
    pub slots: &'a [Option<HotbarSlot>],
    /// The cell in hand.
    pub selected: usize,
}

impl std::fmt::Debug for HudContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("HudContext")
            .field("selected", &self.selected)
            .finish_non_exhaustive()
    }
}

/// The HUD, which remembers when what is in hand changed.
#[derive(Clone, Debug, Default)]
pub struct Hud {
    selected: Option<usize>,
    /// The block in the selected cell, by id.
    held: Option<String>,
    changed_at: f64,
}

impl Hud {
    /// Draws the HUD under whatever is drawn after it, such as a menu.
    pub fn show(&mut self, ui: &mut Ui, context: HudContext<'_>) {
        let now = ui.input(|input| input.time);
        // Taking another cell in hand changes what is in it, and so does
        // putting another block in the cell that is in hand.
        let held = match context.slots.get(context.selected) {
            Some(Some(slot)) => Some(slot.block.as_str()),
            _ => None,
        };
        if self.selected != Some(context.selected) || self.held.as_deref() != held {
            self.selected = Some(context.selected);
            self.held = held.map(str::to_owned);
            self.changed_at = now;
        }
        if context.slots.is_empty() {
            return;
        }
        let screen = ui.ctx().content_rect();
        let painter = ui.painter();
        let count = context.slots.len() as f32;
        let padding = EDGE + 6.0;
        let size = vec2(
            count * SLOT + (count - 1.0) * GAP + 2.0 * padding,
            SLOT + 2.0 * padding,
        );
        let bar = Rect::from_center_size(
            pos2(screen.center().x, screen.max.y - BOTTOM - size.y / 2.0),
            size,
        )
        .round_to_pixels(ui.pixels_per_point());
        panel(painter, context.images, bar);

        for (index, slot) in context.slots.iter().enumerate() {
            let cell = Rect::from_min_size(
                pos2(
                    bar.min.x + padding + index as f32 * (SLOT + GAP),
                    bar.min.y + padding,
                ),
                vec2(SLOT, SLOT),
            );
            let state = if index == context.selected {
                CellState::Selected
            } else {
                CellState::Plain
            };
            let icon = slot.as_ref().and_then(|slot| slot.icon.as_ref());
            paint_cell(painter, cell, icon, Some(index), state);
        }

        // The name of the block in hand, which fades a moment after it
        // changes.
        let age = now - self.changed_at;
        let opacity = if age < NAME_SHOWN {
            1.0
        } else {
            (1.0 - (age - NAME_SHOWN) / NAME_FADE).clamp(0.0, 1.0) as f32
        };
        if opacity > 0.0
            && let Some(Some(slot)) = context.slots.get(context.selected)
        {
            let name = block_name(context.i18n, &slot.block);
            let above = pos2(screen.center().x, bar.min.y - 30.0);
            pixel_text(painter, above, &name, 24.0, Color32::WHITE, true, opacity);
        }
    }
}

/// The name of the block `id`, such as `base:stone`, in the player's
/// language. Its message is `block-stone` for the base game's blocks and
/// `block-<namespace>-<name>` for those of others, so that equal names in
/// different namespaces stay apart.
pub(crate) fn block_name(i18n: &I18n, id: &str) -> String {
    let key = match id.split_once(':') {
        Some((BASE_GAME, path)) | Some(("", path)) => format!("block-{path}"),
        Some((namespace, path)) => format!("block-{namespace}-{path}"),
        None => format!("block-{id}"),
    };
    i18n.get(&key.replace('/', "-"))
}

/// The namespace of the base game's content.
const BASE_GAME: &str = "base";

/// How a cell is drawn.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum CellState {
    Plain,
    /// The pointer is on it, or a block is held over it.
    Hovered,
    /// The one in hand.
    Selected,
}

/// A cell holding a block's picture, with the number of the key that picks
/// it if it is a hotbar cell.
pub(crate) fn paint_cell(
    painter: &egui::Painter,
    cell: Rect,
    icon: Option<&TextureHandle>,
    key: Option<usize>,
    state: CellState,
) {
    // The glow goes under the cell, so it shows only around it.
    if state == CellState::Selected {
        painter.add(
            Shadow {
                offset: [0, 0],
                blur: 20,
                spread: 2,
                color: ACCENT.gamma_multiply(0.45),
            }
            .as_shape(cell, 0.0),
        );
    }
    sunken(painter, cell);
    match state {
        CellState::Plain => {}
        CellState::Hovered => {
            painter.rect_filled(cell, 0.0, Color32::from_white_alpha(22));
            let outline = Stroke::new(EDGE, ACCENT_LIGHT.gamma_multiply(0.55));
            painter.rect_stroke(cell, 0.0, outline, StrokeKind::Outside);
        }
        CellState::Selected => {
            painter.rect_filled(cell, 0.0, ACCENT.gamma_multiply(0.14));
            painter.rect_stroke(cell, 0.0, Stroke::new(EDGE, ACCENT), StrokeKind::Outside);
        }
    }
    if let Some(icon) = icon {
        let at = Rect::from_center_size(cell.center(), vec2(ICON, ICON));
        let uv = Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0));
        painter.image(icon.id(), at, uv, Color32::WHITE);
    }
    if let Some(index) = key {
        let label = ((index + 1) % 10).to_string();
        let corner = cell.min + vec2(TEXEL * 3.0, TEXEL * 2.0);
        for (offset, color) in [
            (TEXEL * 0.5, Color32::from_black_alpha(190)),
            (0.0, Color32::from_white_alpha(150)),
        ] {
            painter.text(
                corner + vec2(offset, offset),
                Align2::LEFT_TOP,
                &label,
                pixel(16.0),
                color,
            );
        }
    }
}

#[cfg(test)]
mod tests {
    use egui::{Context, RawInput};

    use super::*;
    use crate::Language;

    /// Shows the HUD for a frame at `time`.
    fn frame(
        hud: &mut Hud,
        ctx: &Context,
        slots: &[Option<HotbarSlot>],
        selected: usize,
        time: f64,
    ) {
        let i18n = I18n::new(Language::English);
        let images = Images::default();
        let context = HudContext {
            i18n: &i18n,
            images: &images,
            slots,
            selected,
        };
        let input = RawInput {
            screen_rect: Some(Rect::from_min_size(egui::Pos2::ZERO, vec2(1280.0, 720.0))),
            time: Some(time),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| hud.show(ui, context));
        output.textures_delta.clear();
    }

    fn slot(name: &str) -> Option<HotbarSlot> {
        Some(HotbarSlot {
            block: format!("base:{name}"),
            icon: None,
        })
    }

    #[test]
    fn the_name_shows_again_when_the_block_in_hand_changes_by_any_means() {
        let ctx = Context::default();
        crate::style::apply(&ctx);
        let mut hud = Hud::default();
        let mut slots = vec![slot("stone"), slot("dirt"), None];
        frame(&mut hud, &ctx, &slots, 0, 1.0);
        assert_eq!(hud.changed_at, 1.0);
        // Nothing changed: the name goes on fading.
        frame(&mut hud, &ctx, &slots, 0, 5.0);
        assert_eq!(hud.changed_at, 1.0);
        // Another cell in hand.
        frame(&mut hud, &ctx, &slots, 1, 6.0);
        assert_eq!(hud.changed_at, 6.0);
        // Another block put in the cell that is in hand.
        slots[1] = slot("torch");
        frame(&mut hud, &ctx, &slots, 1, 9.0);
        assert_eq!(hud.changed_at, 9.0);
        // The cell emptied.
        slots[1] = None;
        frame(&mut hud, &ctx, &slots, 1, 12.0);
        assert_eq!(hud.changed_at, 12.0);
    }

    #[test]
    fn blocks_are_named_by_namespace_and_path() {
        let english = I18n::new(Language::English);
        assert_eq!(block_name(&english, "base:stone"), "Stone");
        assert_eq!(block_name(&english, "base:copper_ore"), "Copper ore");
        // Another namespace has its own messages, here missing, so the key
        // shows: a block of the same name does not borrow the base game's.
        assert_eq!(
            block_name(&english, "othermod:stone"),
            "block-othermod-stone"
        );
    }
}
