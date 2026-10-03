//! The language screen: the languages there are, one to pick, as in
//! Minecraft, with a choice to follow the system's, and a button to go back.

use egui::{Align2, Rect, Sense, Ui, UiBuilder, Vec2, emath::GuiRounding, pos2, vec2};

use crate::menu::{Menu, MenuAction, MenuContext, background, dim};
use crate::settings::Settings;
use crate::widgets::{BUTTON_GAP, BUTTON_SIZE, PANEL_MARGIN, heading, panel, stone_button_lit};
use crate::{Language, widgets};

/// Between the buttons.
const GAP: f32 = BUTTON_GAP;
/// What the title takes, before the first button.
const TITLE: f32 = 56.0;

/// What can be picked: following the system's language, or one of the ones
/// there are.
fn choices() -> Vec<Option<Language>> {
    std::iter::once(None)
        .chain(Language::ALL.into_iter().map(Some))
        .collect()
}

/// How large the panel is, for `count` choices: the title, a button for
/// each, a gap, and the button to go back.
fn panel_size(count: usize) -> Vec2 {
    let languages = count as f32 * BUTTON_SIZE.y + (count as f32 - 1.0).max(0.0) * GAP;
    vec2(
        BUTTON_SIZE.x + 2.0 * PANEL_MARGIN,
        TITLE + GAP + languages + 2.0 * GAP + BUTTON_SIZE.y + 2.0 * PANEL_MARGIN,
    )
}

/// Where the button of the `index`th choice is, in a panel at `panel`.
fn language_button(panel: Rect, index: usize) -> Rect {
    let inner = panel.shrink(PANEL_MARGIN);
    Rect::from_min_size(
        inner.min + vec2(0.0, TITLE + GAP + index as f32 * (BUTTON_SIZE.y + GAP)),
        BUTTON_SIZE,
    )
}

/// Where the button to go back is.
fn back_button(panel: Rect) -> Rect {
    let inner = panel.shrink(PANEL_MARGIN);
    Rect::from_min_size(pos2(inner.min.x, inner.max.y - BUTTON_SIZE.y), BUTTON_SIZE)
}

impl Menu {
    pub(crate) fn language_screen(
        &mut self,
        ui: &mut Ui,
        context: MenuContext<'_>,
        settings: &mut Settings,
    ) -> Option<MenuAction> {
        let (images, i18n) = (context.images, context.i18n);
        background(ui, images, context.world_visible);
        dim(ui);
        let choices = choices();
        let mut done = false;
        egui::Area::new("language".into())
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                let (rect, _) = ui.allocate_exact_size(panel_size(choices.len()), Sense::hover());
                let rect = rect.round_to_pixels(ui.pixels_per_point());
                panel(ui.painter(), images, rect);
                let inner = rect.shrink(PANEL_MARGIN);
                heading(
                    ui.painter(),
                    pos2(inner.center().x, inner.min.y + TITLE / 2.0),
                    &i18n.get("settings-language"),
                );
                for (index, choice) in choices.iter().enumerate() {
                    let at = language_button(rect, index);
                    ui.scope_builder(UiBuilder::new().max_rect(at), |ui| {
                        let lit = *choice == settings.language;
                        let name = match choice {
                            Some(language) => language.native_name().to_owned(),
                            None => i18n.get("language-system"),
                        };
                        if stone_button_lit(ui, images, &name, BUTTON_SIZE, true, lit).clicked() {
                            settings.language = *choice;
                        }
                    });
                }
                let at = back_button(rect);
                ui.scope_builder(UiBuilder::new().max_rect(at), |ui| {
                    let name = i18n.get("settings-back");
                    done = widgets::stone_button(ui, images, &name, true).clicked();
                });
            });
        if done { self.back() } else { None }
    }
}

#[cfg(test)]
mod tests {
    use egui::{Context, Event, Modifiers, PointerButton, Pos2, RawInput};

    use super::*;
    use crate::widgets::Images;
    use crate::{InventoryContext, Screen};

    struct Table {
        ctx: Context,
        menu: Menu,
        settings: Settings,
        i18n: crate::I18n,
        images: Images,
        time: f64,
    }

    impl Table {
        fn new() -> Self {
            let ctx = Context::default();
            crate::style::apply(&ctx);
            Self {
                ctx,
                menu: Menu::new(Screen::Language),
                settings: Settings::default(),
                i18n: crate::I18n::new(Language::English),
                images: Images::default(),
                time: 0.0,
            }
        }

        fn screen() -> Rect {
            Rect::from_min_size(Pos2::ZERO, vec2(1280.0, 800.0))
        }

        fn panel() -> Rect {
            Rect::from_center_size(Self::screen().center(), panel_size(choices().len()))
        }

        fn frame(&mut self, events: Vec<Event>) -> Option<MenuAction> {
            self.time += 0.05;
            let input = RawInput {
                screen_rect: Some(Self::screen()),
                time: Some(self.time),
                events,
                ..Default::default()
            };
            let context = MenuContext {
                i18n: &self.i18n,
                images: &self.images,
                inventory: InventoryContext {
                    catalog: &[],
                    hotbar: &[],
                    selected: 0,
                },
                max_scale: 100,
                world_visible: false,
                version: "test",
                system_language: Language::English,
            };
            let (menu, settings) = (&mut self.menu, &mut self.settings);
            let mut action = None;
            let mut output = self.ctx.run_ui(input, |ui| {
                action = menu.show(ui, context, settings);
            });
            output.textures_delta.clear();
            action
        }

        fn click(&mut self, at: Pos2) -> Option<MenuAction> {
            self.frame(vec![Event::PointerMoved(at)]);
            for pressed in [true, false] {
                let action = self.frame(vec![Event::PointerButton {
                    pos: at,
                    button: PointerButton::Primary,
                    pressed,
                    modifiers: Modifiers::NONE,
                }]);
                if !pressed {
                    return action;
                }
            }
            None
        }
    }

    #[test]
    fn clicking_a_language_picks_it_and_the_button_back_goes_to_the_menu() {
        let mut table = Table::new();
        table.frame(vec![]);
        for (index, choice) in choices().into_iter().enumerate() {
            // Start from the other state, so that the click has to change it.
            table.settings.language = choice.map_or(Some(Language::English), |_| None);
            table.click(language_button(Table::panel(), index).center());
            assert_eq!(table.settings.language, choice);
        }
        // Back to the menu, and the settings are worth saving.
        let action = table.click(back_button(Table::panel()).center());
        assert_eq!(action, Some(MenuAction::SettingsClosed));
        assert_eq!(table.menu.screen, Screen::Main);
    }

    #[test]
    fn escape_leaves_the_language_screen_too() {
        let mut menu = Menu::new(Screen::Language);
        assert_eq!(menu.back(), Some(MenuAction::SettingsClosed));
        assert_eq!(menu.screen, Screen::Main);
    }

    #[test]
    fn the_panel_holds_every_language_and_the_way_back() {
        let panel = Table::panel();
        for index in 0..choices().len() {
            assert!(panel.contains_rect(language_button(panel, index)));
        }
        let back = back_button(panel);
        assert!(panel.contains_rect(back));
        // Nothing overlaps.
        let last = language_button(panel, choices().len() - 1);
        assert!(back.min.y >= last.max.y);
    }
}
