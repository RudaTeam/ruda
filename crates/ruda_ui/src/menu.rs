use egui::{Align2, Color32, ComboBox, RichText, Slider, TextureHandle, Ui, Vec2};

use crate::settings::{
    Clouds, FIELDS_OF_VIEW, FpsLimit, GpuApi, LOD_DISTANCES, Lighting, Preset, Settings,
    VIEW_DISTANCES,
};
use crate::{I18n, Language};

const BUTTON_SIZE: Vec2 = Vec2::new(340.0, 48.0);
/// Fits the longest label and value in either language.
const SETTINGS_WIDTH: f32 = 800.0;
const COMBO_WIDTH: f32 = 300.0;

/// Which menu is open.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Screen {
    Main,
    /// Over a running game.
    Paused,
    Settings {
        /// Where "Back" leads.
        from_game: bool,
    },
}

/// Something the menu needs the game to do.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MenuAction {
    StartSingleplayer,
    Resume,
    QuitToTitle,
    Exit,
    /// The settings screen closed; a good moment to save them.
    SettingsClosed,
}

/// Things the menu shows but does not own.
#[derive(Clone, Copy)]
pub struct MenuContext<'a> {
    pub i18n: &'a I18n,
    pub logo: Option<&'a TextureHandle>,
    pub version: &'a str,
    /// Language used when the settings leave it to the system.
    pub system_language: Language,
}

impl std::fmt::Debug for MenuContext<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("MenuContext")
            .field("version", &self.version)
            .field("system_language", &self.system_language)
            .finish_non_exhaustive()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Menu {
    pub screen: Screen,
}

impl Menu {
    pub fn new(screen: Screen) -> Self {
        Self { screen }
    }

    /// Draws the open screen. Settings change in place as the player edits
    /// them.
    pub fn show(
        &mut self,
        ui: &mut Ui,
        context: MenuContext<'_>,
        settings: &mut Settings,
    ) -> Option<MenuAction> {
        match self.screen {
            Screen::Main => self.main(ui, context),
            Screen::Paused => self.paused(ui, context),
            Screen::Settings { .. } => self.settings(ui, context, settings),
        }
    }

    /// What Escape does: leaves the settings or resumes the game.
    pub fn back(&mut self) -> Option<MenuAction> {
        match self.screen {
            Screen::Main => None,
            Screen::Paused => Some(MenuAction::Resume),
            Screen::Settings { from_game } => {
                self.screen = if from_game {
                    Screen::Paused
                } else {
                    Screen::Main
                };
                Some(MenuAction::SettingsClosed)
            }
        }
    }

    fn main(&mut self, ui: &mut Ui, context: MenuContext<'_>) -> Option<MenuAction> {
        let t = |id| context.i18n.get(id);
        let mut action = None;
        centered(ui, "main menu", |ui| {
            if let Some(logo) = context.logo {
                let width = (ui.ctx().content_rect().width() * 0.5).clamp(320.0, 640.0);
                ui.add(egui::Image::new(logo).max_width(width));
                ui.add_space(24.0);
            }
            if button(ui, &t("menu-singleplayer")).clicked() {
                action = Some(MenuAction::StartSingleplayer);
            }
            ui.add_enabled(
                false,
                egui::Button::new(t("menu-multiplayer")).min_size(BUTTON_SIZE),
            )
            .on_disabled_hover_text(t("menu-coming-soon"));
            if button(ui, &t("menu-settings")).clicked() {
                self.screen = Screen::Settings { from_game: false };
            }
            if button(ui, &t("menu-exit")).clicked() {
                action = Some(MenuAction::Exit);
            }
        });
        egui::Area::new("version".into())
            .anchor(Align2::LEFT_BOTTOM, [12.0, -10.0])
            .show(ui.ctx(), |ui| {
                ui.label(
                    RichText::new(format!("Ruda {}", context.version))
                        .small()
                        .weak(),
                );
            });
        action
    }

    fn paused(&mut self, ui: &mut Ui, context: MenuContext<'_>) -> Option<MenuAction> {
        let t = |id| context.i18n.get(id);
        dim(ui);
        let mut action = None;
        centered(ui, "pause menu", |ui| {
            ui.heading(t("menu-paused"));
            ui.add_space(12.0);
            if button(ui, &t("menu-resume")).clicked() {
                action = Some(MenuAction::Resume);
            }
            if button(ui, &t("menu-settings")).clicked() {
                self.screen = Screen::Settings { from_game: true };
            }
            if button(ui, &t("menu-quit-to-title")).clicked() {
                action = Some(MenuAction::QuitToTitle);
            }
        });
        action
    }

    fn settings(
        &mut self,
        ui: &mut Ui,
        context: MenuContext<'_>,
        settings: &mut Settings,
    ) -> Option<MenuAction> {
        let i18n = context.i18n;
        let t = |id| i18n.get(id);
        if matches!(self.screen, Screen::Settings { from_game: true }) {
            dim(ui);
        }
        let mut action = None;
        centered(ui, "settings", |ui| {
            egui::Frame::window(ui.style())
                .inner_margin(24.0)
                .show(ui, |ui| {
                    ui.set_width(SETTINGS_WIDTH);
                    ui.vertical_centered(|ui| ui.heading(t("settings-title")));
                    ui.with_layout(egui::Layout::top_down(egui::Align::Min), |ui| {
                        ui.add_space(8.0);

                        let heading = |ui: &mut Ui, id| {
                            ui.label(
                                RichText::new(t(id))
                                    .strong()
                                    .color(Color32::from_rgb(232, 128, 48)),
                            )
                        };
                        let graphics = &mut settings.graphics;
                        heading(ui, "settings-graphics");
                        egui::Grid::new("graphics")
                            .num_columns(2)
                            .spacing([24.0, 14.0])
                            .show(ui, |ui| {
                                ui.label(t("settings-preset"));
                                let preset_name = |preset: Option<Preset>| {
                                    t(match preset {
                                        Some(Preset::Standard) => "settings-preset-standard",
                                        Some(Preset::High) => "settings-preset-high",
                                        Some(Preset::Ultra) => "settings-preset-ultra",
                                        None => "settings-preset-custom",
                                    })
                                };
                                let current = Preset::of(graphics);
                                ComboBox::from_id_salt("preset")
                                    .selected_text(preset_name(current))
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for preset in Preset::ALL {
                                            let chosen = ui.selectable_label(
                                                current == Some(preset),
                                                preset_name(Some(preset)),
                                            );
                                            if chosen.clicked() {
                                                preset.apply(graphics);
                                            }
                                        }
                                    });
                                ui.end_row();

                                ui.label(t("settings-lighting"));
                                let lighting_name = |lighting: Lighting| {
                                    t(match lighting {
                                        Lighting::Classic => "settings-lighting-classic",
                                        Lighting::Atmospheric => "settings-lighting-atmospheric",
                                    })
                                };
                                ComboBox::from_id_salt("lighting")
                                    .selected_text(lighting_name(graphics.lighting))
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for lighting in Lighting::ALL {
                                            ui.selectable_value(
                                                &mut graphics.lighting,
                                                lighting,
                                                lighting_name(lighting),
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label(t("settings-view-distance"));
                                ui.horizontal(|ui| {
                                    ui.add(
                                        Slider::new(&mut graphics.view_distance, VIEW_DISTANCES)
                                            .show_value(false),
                                    );
                                    let chunks = i64::from(graphics.view_distance);
                                    ui.label(i18n.get_with(
                                        "settings-view-distance-value",
                                        &[("chunks", chunks), ("blocks", chunks * 32)],
                                    ));
                                });
                                ui.end_row();

                                ui.label(t("settings-fov"));
                                ui.horizontal(|ui| {
                                    ui.add(
                                        Slider::new(&mut graphics.fov, FIELDS_OF_VIEW)
                                            .show_value(false),
                                    );
                                    let degrees = i64::from(graphics.fov);
                                    ui.label(
                                        i18n.get_with(
                                            "settings-fov-value",
                                            &[("degrees", degrees)],
                                        ),
                                    );
                                });
                                ui.end_row();

                                ui.label(t("settings-view-bobbing"));
                                ui.checkbox(&mut graphics.view_bobbing, "");
                                ui.end_row();

                                ui.label(t("settings-fps-limit"));
                                let limit_name = |limit: FpsLimit| match limit.fps() {
                                    Some(fps) => i18n.get_with(
                                        "settings-fps-limit-value",
                                        &[("fps", i64::from(fps))],
                                    ),
                                    None if limit == FpsLimit::Display => {
                                        t("settings-fps-limit-display")
                                    }
                                    None => t("settings-fps-limit-off"),
                                };
                                ComboBox::from_id_salt("fps limit")
                                    .selected_text(limit_name(graphics.fps_limit))
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for limit in FpsLimit::ALL {
                                            ui.selectable_value(
                                                &mut graphics.fps_limit,
                                                limit,
                                                limit_name(limit),
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label(t("settings-fullscreen"));
                                ui.checkbox(&mut graphics.fullscreen, "");
                                ui.end_row();

                                ui.label(t("settings-clouds"));
                                let clouds_name = |clouds: Clouds| {
                                    t(match clouds {
                                        Clouds::Off => "settings-clouds-off",
                                        Clouds::Standard => "settings-clouds-standard",
                                        Clouds::Volumetric => "settings-clouds-volumetric",
                                    })
                                };
                                ComboBox::from_id_salt("clouds")
                                    .selected_text(clouds_name(graphics.clouds))
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for clouds in Clouds::ALL {
                                            ui.selectable_value(
                                                &mut graphics.clouds,
                                                clouds,
                                                clouds_name(clouds),
                                            );
                                        }
                                    })
                                    .response
                                    .on_hover_text(t("settings-clouds-note"));
                                ui.end_row();

                                ui.label(t("settings-shadows"));
                                ui.checkbox(&mut graphics.shadows, "")
                                    .on_hover_text(t("settings-shadows-note"));
                                ui.end_row();

                                ui.label(t("settings-lod"));
                                let lod_name = |blocks: u16| {
                                    if blocks == 0 {
                                        t("settings-lod-off")
                                    } else {
                                        i18n.get_with(
                                            "settings-lod-value",
                                            &[("blocks", i64::from(blocks))],
                                        )
                                    }
                                };
                                ComboBox::from_id_salt("lod")
                                    .selected_text(lod_name(graphics.lod_distance))
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for blocks in LOD_DISTANCES {
                                            ui.selectable_value(
                                                &mut graphics.lod_distance,
                                                blocks,
                                                lod_name(blocks),
                                            );
                                        }
                                    });
                                ui.end_row();

                                ui.label(t("settings-gpu-api"));
                                let api_name = |api: GpuApi| {
                                    api.name()
                                        .map_or_else(|| t("settings-gpu-api-auto"), str::to_owned)
                                };
                                ComboBox::from_id_salt("gpu api")
                                    .selected_text(api_name(graphics.gpu_api))
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for &api in GpuApi::available() {
                                            ui.selectable_value(
                                                &mut graphics.gpu_api,
                                                api,
                                                api_name(api),
                                            );
                                        }
                                    });
                                ui.end_row();
                                ui.label("");
                                ui.label(RichText::new(t("settings-restart-note")).small().weak());
                                ui.end_row();

                                ui.label(t("settings-language"));
                                let current = settings.language.unwrap_or(context.system_language);
                                ComboBox::from_id_salt("language")
                                    .selected_text(current.native_name())
                                    .width(COMBO_WIDTH)
                                    .show_ui(ui, |ui| {
                                        for language in Language::ALL {
                                            if ui
                                                .selectable_label(
                                                    current == language,
                                                    language.native_name(),
                                                )
                                                .clicked()
                                            {
                                                settings.language = Some(language);
                                            }
                                        }
                                    });
                                ui.end_row();

                                heading(ui, "settings-controls");
                                ui.end_row();
                                ui.label(t("settings-auto-jump"));
                                ui.checkbox(&mut settings.controls.auto_jump, "")
                                    .on_hover_text(t("settings-auto-jump-note"));
                                ui.end_row();
                            });
                    });

                    ui.add_space(12.0);
                    ui.vertical_centered(|ui| {
                        if button(ui, &t("settings-back")).clicked() {
                            action = self.back();
                        }
                    });
                });
        });
        action
    }
}

fn button(ui: &mut Ui, text: &str) -> egui::Response {
    ui.add(egui::Button::new(text).min_size(BUTTON_SIZE))
}

/// Lays out `contents` in a column centred on the screen.
fn centered(ui: &mut Ui, id: &str, contents: impl FnOnce(&mut Ui)) {
    egui::Area::new(egui::Id::new(id))
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ui.ctx(), |ui| {
            ui.vertical_centered(contents);
        });
}

/// Darkens the game behind a menu.
fn dim(ui: &mut Ui) {
    let screen = ui.ctx().content_rect();
    ui.painter()
        .rect_filled(screen, 0.0, Color32::from_black_alpha(150));
}
