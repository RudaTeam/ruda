//! The settings screen: tabs on the left, the settings of the open tab on
//! the right and a line of help for the one under the pointer below.

use egui::{
    Align, Align2, Color32, Layout, Rect, Response, RichText, ScrollArea, Sense, Ui, UiBuilder,
    Vec2, emath::GuiRounding, pos2, scroll_area::ScrollBarVisibility, vec2,
};

use crate::controls::{CONTROL_SIZE, TAB_SIZE, cycle, slider, tab};
use crate::menu::{Menu, MenuAction, MenuContext, Screen, background, dim};
use crate::settings::{
    Appearance, ControlAction, Controls, FIELDS_OF_VIEW, FpsLimit, GpuApi, Graphics, LOD_DISTANCES,
    Lighting, MOUSE_SENSITIVITIES, Preset, Settings, Shadows, UI_SCALE_STEP, UI_SCALES,
    VIEW_DISTANCES,
};
use crate::style::{ACCENT_LIGHT, TEXT, pixel};
use crate::widgets::{
    EDGE, Images, PANEL_MARGIN, TEXEL, heading, panel, stone_button_sized, sunken,
};
use crate::{I18n, Language};

const PANEL_SIZE: Vec2 = Vec2::new(1000.0, 800.0);
/// Room for the longest name of a setting, in either language.
const LABEL_WIDTH: f32 = 330.0;
const ROW_GAP: f32 = 8.0;
const FOOTER_BUTTON: Vec2 = Vec2::new(220.0, 48.0);

/// A page of the settings.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Tab {
    #[default]
    Graphics,
    Controls,
    Interface,
}

impl Tab {
    const ALL: [Tab; 3] = [Tab::Graphics, Tab::Controls, Tab::Interface];

    fn name(self) -> &'static str {
        match self {
            Tab::Graphics => "settings-graphics",
            Tab::Controls => "settings-controls",
            Tab::Interface => "settings-interface",
        }
    }
}

/// What the rows of a page share.
struct Page<'a> {
    images: &'a Images,
    i18n: &'a I18n,
}

impl Page<'_> {
    /// A setting: its name, and what sets it.
    fn row<R>(
        &mut self,
        ui: &mut Ui,
        name: &str,
        control: impl FnOnce(&mut Ui, &Images, &I18n) -> (R, Response),
    ) -> R {
        let (images, i18n) = (self.images, self.i18n);
        let row = ui.horizontal(|ui| {
            ui.set_height(CONTROL_SIZE.y);
            // A column of fixed width, so every control starts at the same
            // place whatever its name is.
            let (label, _) =
                ui.allocate_exact_size(vec2(LABEL_WIDTH, CONTROL_SIZE.y), Sense::hover());
            // The names are in the pixel font, like the buttons they sit by.
            let name = i18n.get(name);
            for (offset, color) in [(TEXEL, Color32::from_black_alpha(166)), (0.0, TEXT)] {
                ui.painter().text(
                    label.left_center() + vec2(offset, offset),
                    Align2::LEFT_CENTER,
                    &name,
                    pixel(24.0),
                    color,
                );
            }
            control(ui, images, i18n)
        });
        row.inner.0
    }
}

/// Steps through `options` from `value`, a step either way, wrapping
/// round.
fn step_through<T: Copy + PartialEq>(options: &[T], value: Option<T>, step: i32) -> T {
    let last = options.len() as i32 - 1;
    let at = value.and_then(|value| options.iter().position(|&option| option == value));
    let next = match at {
        Some(at) => (at as i32 + step).rem_euclid(last + 1),
        None if step > 0 => 0,
        None => last,
    };
    options[next as usize]
}

/// A setting of a few named values, stepped through by a click.
fn pick<T: Copy + PartialEq>(
    ui: &mut Ui,
    images: &Images,
    options: &[T],
    value: &mut T,
    name: impl Fn(T) -> String,
) -> ((), Response) {
    let (step, response) = cycle(ui, images, &name(*value), None);
    if step != 0 {
        *value = step_through(options, Some(*value), step);
    }
    ((), response)
}

fn toggle(ui: &mut Ui, images: &Images, i18n: &I18n, value: &mut bool) -> ((), Response) {
    let (text, color) = if *value {
        (i18n.get("settings-on"), ACCENT_LIGHT)
    } else {
        (i18n.get("settings-off"), TEXT.gamma_multiply(0.55))
    };
    let (step, response) = cycle(ui, images, &text, Some(color));
    if step != 0 {
        *value = !*value;
    }
    ((), response)
}

/// A slider over whole numbers.
fn number(
    ui: &mut Ui,
    images: &Images,
    value: &mut u8,
    range: &std::ops::RangeInclusive<u8>,
    step: u8,
    text: &str,
) -> ((), Response) {
    let mut float = f32::from(*value);
    let response = slider(
        ui,
        images,
        &mut float,
        f32::from(*range.start())..=f32::from(*range.end()),
        f32::from(step),
        text,
    );
    *value = float.round() as u8;
    ((), response)
}

fn percent(i18n: &I18n, value: u8) -> String {
    i18n.get_with("settings-percent", &[("percent", i64::from(value))])
}

/// What a button is called, from its name in the input.
fn button_label(i18n: &I18n, name: &str) -> String {
    let special = match name {
        "mouse:Left" => "key-mouse-left",
        "mouse:Right" => "key-mouse-right",
        "mouse:Middle" => "key-mouse-middle",
        "mouse:Back" => "key-mouse-back",
        "mouse:Forward" => "key-mouse-forward",
        "Space" => "key-space",
        "ShiftLeft" => "key-shift-left",
        "ShiftRight" => "key-shift-right",
        "ControlLeft" => "key-control-left",
        "ControlRight" => "key-control-right",
        "AltLeft" => "key-alt-left",
        "AltRight" => "key-alt-right",
        "Tab" => "key-tab",
        "Enter" => "key-enter",
        "Backspace" => "key-backspace",
        "CapsLock" => "key-caps-lock",
        "ArrowUp" => "key-arrow-up",
        "ArrowDown" => "key-arrow-down",
        "ArrowLeft" => "key-arrow-left",
        "ArrowRight" => "key-arrow-right",
        _ => "",
    };
    if !special.is_empty() {
        return i18n.get(special);
    }
    if let Some(letter) = name.strip_prefix("Key") {
        return letter.to_owned();
    }
    if let Some(digit) = name.strip_prefix("Digit") {
        return digit.to_owned();
    }
    if let Some(number) = name.strip_prefix("Numpad") {
        return format!("Num {number}");
    }
    name.to_owned()
}

fn action_name(action: ControlAction) -> &'static str {
    match action {
        ControlAction::MoveForward => "control-move-forward",
        ControlAction::MoveBack => "control-move-back",
        ControlAction::MoveLeft => "control-move-left",
        ControlAction::MoveRight => "control-move-right",
        ControlAction::Jump => "control-jump",
        ControlAction::Sneak => "control-sneak",
        ControlAction::Sprint => "control-sprint",
        ControlAction::Break => "control-break",
        ControlAction::Place => "control-place",
        ControlAction::Inventory => "control-inventory",
    }
}

impl Menu {
    pub(crate) fn settings_screen(
        &mut self,
        ui: &mut Ui,
        context: MenuContext<'_>,
        settings: &mut Settings,
    ) -> Option<MenuAction> {
        let images = context.images;
        if !matches!(self.screen, Screen::Settings { from_game: true }) {
            background(ui, images, context.world_visible);
        }
        dim(ui);

        let screen = ui.ctx().content_rect();
        let size = vec2(
            PANEL_SIZE.x.min(screen.width() - 48.0),
            PANEL_SIZE.y.min(screen.height() - 48.0),
        );
        let mut page = Page {
            images,
            i18n: context.i18n,
        };
        let (mut back, mut reset) = (false, false);
        egui::Area::new("settings".into())
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                let (rect, _) = ui.allocate_exact_size(size, Sense::hover());
                let rect = rect.round_to_pixels(ui.pixels_per_point());
                panel(ui.painter(), images, rect);

                let inner = rect.shrink(PANEL_MARGIN);
                let title = Rect::from_min_size(inner.min, vec2(inner.width(), 48.0));
                let footer = Rect::from_min_max(pos2(inner.min.x, inner.max.y - 56.0), inner.max);
                let body = Rect::from_min_max(
                    pos2(inner.min.x, title.max.y + 12.0),
                    pos2(inner.max.x, footer.min.y - 16.0),
                );
                let tabs = Rect::from_min_size(body.min, vec2(TAB_SIZE.x, body.height()));
                let content = Rect::from_min_max(pos2(tabs.max.x + 20.0, body.min.y), body.max);

                heading(
                    ui.painter(),
                    title.center(),
                    &context.i18n.get("settings-title"),
                );

                ui.scope_builder(UiBuilder::new().max_rect(tabs), |ui| {
                    ui.spacing_mut().item_spacing.y = ROW_GAP;
                    for open in Tab::ALL {
                        let name = context.i18n.get(open.name());
                        if tab(ui, &name, self.tab == open, true).clicked() {
                            self.tab = open;
                            self.listening = None;
                        }
                    }
                    let sound = context.i18n.get("settings-sound");
                    tab(ui, &sound, false, false).on_hover_text(context.i18n.get("settings-soon"));
                });

                sunken(ui.painter(), content);
                ui.scope_builder(
                    UiBuilder::new().max_rect(content.shrink(EDGE + 14.0)),
                    |ui| {
                        // A bar with a body, not a thin line that shows on hover.
                        ui.spacing_mut().scroll = egui::style::ScrollStyle::solid();
                        ScrollArea::vertical()
                            .auto_shrink(false)
                            .scroll_bar_visibility(ScrollBarVisibility::AlwaysVisible)
                            .show(ui, |ui| {
                                ui.spacing_mut().item_spacing.y = ROW_GAP;
                                match self.tab {
                                    Tab::Graphics => {
                                        graphics_page(ui, &mut page, &mut settings.graphics)
                                    }
                                    Tab::Controls => controls_page(
                                        ui,
                                        &mut page,
                                        &mut settings.controls,
                                        &mut self.listening,
                                    ),
                                    Tab::Interface => interface_page(
                                        ui,
                                        &mut page,
                                        &mut settings.language,
                                        &mut settings.appearance,
                                        context.system_language,
                                        context.max_scale,
                                    ),
                                }
                            });
                    },
                );

                let buttons = Rect::from_center_size(
                    footer.center(),
                    vec2(FOOTER_BUTTON.x * 2.0 + 24.0, FOOTER_BUTTON.y),
                );
                ui.scope_builder(
                    UiBuilder::new()
                        .max_rect(buttons)
                        .layout(Layout::left_to_right(Align::Center)),
                    |ui| {
                        ui.spacing_mut().item_spacing.x = 24.0;
                        let name = context.i18n.get("settings-back");
                        back = stone_button_sized(ui, images, &name, FOOTER_BUTTON, true).clicked();
                        let name = context.i18n.get("settings-reset");
                        reset =
                            stone_button_sized(ui, images, &name, FOOTER_BUTTON, true).clicked();
                    },
                );
            });

        if reset {
            self.listening = None;
            match self.tab {
                Tab::Graphics => settings.graphics = Graphics::default(),
                Tab::Controls => settings.controls = Controls::default(),
                Tab::Interface => {
                    settings.language = None;
                    settings.appearance = Appearance::default();
                }
            }
        }
        if back {
            self.listening = None;
            return self.back();
        }
        None
    }
}

fn graphics_page(ui: &mut Ui, page: &mut Page<'_>, graphics: &mut Graphics) {
    page.row(ui, "settings-preset", |ui, images, t| {
        let name = |preset: Option<Preset>| {
            t.get(match preset {
                Some(Preset::Standard) => "settings-preset-standard",
                Some(Preset::High) => "settings-preset-high",
                Some(Preset::Ultra) => "settings-preset-ultra",
                None => "settings-preset-custom",
            })
        };
        let current = Preset::of(graphics);
        let (step, response) = cycle(ui, images, &name(current), None);
        if step != 0 {
            step_through(&Preset::ALL, current, step).apply(graphics);
        }
        ((), response)
    });
    page.row(ui, "settings-lighting", |ui, images, t| {
        let name = |lighting| {
            t.get(match lighting {
                Lighting::Classic => "settings-lighting-classic",
                Lighting::Atmospheric => "settings-lighting-atmospheric",
            })
        };
        pick(ui, images, &Lighting::ALL, &mut graphics.lighting, name)
    });
    page.row(ui, "settings-view-distance", |ui, images, t| {
        let chunks = i64::from(graphics.view_distance);
        let text = t.get_with(
            "settings-view-distance-value",
            &[("chunks", chunks), ("blocks", chunks * 32)],
        );
        number(
            ui,
            images,
            &mut graphics.view_distance,
            &VIEW_DISTANCES,
            1,
            &text,
        )
    });
    page.row(ui, "settings-lod", |ui, images, t| {
        let name = |blocks: u16| {
            if blocks == 0 {
                t.get("settings-lod-off")
            } else {
                t.get_with("settings-lod-value", &[("blocks", i64::from(blocks))])
            }
        };
        pick(ui, images, &LOD_DISTANCES, &mut graphics.lod_distance, name)
    });
    page.row(ui, "settings-fov", |ui, images, t| {
        let text = t.get_with(
            "settings-fov-value",
            &[("degrees", i64::from(graphics.fov))],
        );
        number(ui, images, &mut graphics.fov, &FIELDS_OF_VIEW, 1, &text)
    });
    page.row(ui, "settings-view-bobbing", |ui, images, t| {
        toggle(ui, images, t, &mut graphics.view_bobbing)
    });
    page.row(ui, "settings-fps-limit", |ui, images, t| {
        let name = |limit: FpsLimit| match limit.fps() {
            Some(fps) => t.get_with("settings-fps-limit-value", &[("fps", i64::from(fps))]),
            None if limit == FpsLimit::Display => t.get("settings-fps-limit-display"),
            None => t.get("settings-fps-limit-off"),
        };
        pick(ui, images, &FpsLimit::ALL, &mut graphics.fps_limit, name)
    });
    page.row(ui, "settings-fullscreen", |ui, images, t| {
        toggle(ui, images, t, &mut graphics.fullscreen)
    });
    page.row(ui, "settings-clouds", |ui, images, t| {
        toggle(ui, images, t, &mut graphics.clouds)
    });
    page.row(ui, "settings-shadows", |ui, images, t| {
        let name = |shadows| {
            t.get(match shadows {
                Shadows::Off => "settings-shadows-off",
                Shadows::Standard => "settings-shadows-standard",
                Shadows::Rays => "settings-shadows-rays",
            })
        };
        pick(ui, images, &Shadows::ALL, &mut graphics.shadows, name)
    });
    page.row(ui, "settings-menu-world", |ui, images, t| {
        toggle(ui, images, t, &mut graphics.menu_world)
    });
    page.row(ui, "settings-gpu-api", |ui, images, t| {
        let name = |api: GpuApi| {
            api.name()
                .map_or_else(|| t.get("settings-gpu-api-auto"), str::to_owned)
        };
        pick(ui, images, GpuApi::available(), &mut graphics.gpu_api, name)
    });
}

fn controls_page(
    ui: &mut Ui,
    page: &mut Page<'_>,
    controls: &mut Controls,
    listening: &mut Option<ControlAction>,
) {
    page.row(ui, "settings-mouse-sensitivity", |ui, images, t| {
        let text = percent(t, controls.mouse_sensitivity);
        number(
            ui,
            images,
            &mut controls.mouse_sensitivity,
            &MOUSE_SENSITIVITIES,
            5,
            &text,
        )
    });
    page.row(ui, "settings-auto-jump", |ui, images, t| {
        toggle(ui, images, t, &mut controls.auto_jump)
    });
    ui.add_space(6.0);
    ui.label(
        RichText::new(page.i18n.get("settings-keys"))
            .font(pixel(24.0))
            .color(ACCENT_LIGHT),
    );
    for action in ControlAction::ALL {
        let picking = *listening == Some(action);
        let clicked = page.row(ui, action_name(action), |ui, images, t| {
            let text = if picking {
                t.get("settings-key-listening")
            } else {
                button_label(t, controls.key(action))
            };
            let response = crate::controls::key_button(ui, images, &text, picking);
            (response.clicked(), response)
        });
        if clicked {
            *listening = if picking { None } else { Some(action) };
        }
    }
}

fn interface_page(
    ui: &mut Ui,
    page: &mut Page<'_>,
    language: &mut Option<Language>,
    appearance: &mut Appearance,
    system_language: Language,
    max_scale: u8,
) {
    page.row(ui, "settings-language", |ui, images, _| {
        let before = language.unwrap_or(system_language);
        let mut current = before;
        let result = pick(ui, images, &Language::ALL, &mut current, |language| {
            language.native_name().to_owned()
        });
        if current != before {
            *language = Some(current);
        }
        result
    });
    page.row(ui, "settings-ui-scale", |ui, images, t| {
        // The sizes that fit the window, never fewer than the smallest. A
        // button, not a slider: the menus change size under the pointer, and
        // a slider under it would chase its own movement.
        let fitting = max_scale.max(*UI_SCALES.start());
        let steps: Vec<u8> = UI_SCALES
            .step_by(usize::from(UI_SCALE_STEP))
            .filter(|&scale| scale <= fitting)
            .collect();
        // What is in force: the size picked, held back by the window.
        let largest = steps.last().copied().unwrap_or(*UI_SCALES.start());
        let before = appearance.scale.min(largest);
        let mut shown = before;
        let result = pick(ui, images, &steps, &mut shown, |scale| percent(t, scale));
        if shown != before {
            appearance.scale = shown;
        }
        result
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn values_step_round_and_start_from_an_end_when_unknown() {
        let options = [10, 20, 30];
        assert_eq!(step_through(&options, Some(10), 1), 20);
        assert_eq!(step_through(&options, Some(30), 1), 10);
        assert_eq!(step_through(&options, Some(10), -1), 30);
        // A value that is not one of them, such as a custom preset.
        assert_eq!(step_through(&options, None, 1), 10);
        assert_eq!(step_through(&options, None, -1), 30);
    }

    #[test]
    fn buttons_are_named_for_people() {
        let english = I18n::new(Language::English);
        let russian = I18n::new(Language::Russian);
        assert_eq!(button_label(&english, "KeyW"), "W");
        assert_eq!(button_label(&english, "Digit5"), "5");
        assert_eq!(button_label(&english, "Space"), "Space");
        assert_eq!(button_label(&russian, "Space"), "Пробел");
        assert_eq!(button_label(&russian, "mouse:Right"), "Правая кнопка мыши");
        // What has no name of its own shows as the input knows it.
        assert_eq!(button_label(&english, "F5"), "F5");
    }

    #[test]
    fn every_action_and_page_has_a_name_in_every_language() {
        for language in Language::ALL {
            let i18n = I18n::new(language);
            for action in ControlAction::ALL {
                let name = action_name(action);
                assert_ne!(i18n.get(name), name, "{language:?} {name}");
            }
            for tab in Tab::ALL {
                assert_ne!(i18n.get(tab.name()), tab.name(), "{language:?}");
            }
        }
    }
}
