use egui::emath::GuiRounding;
use egui::{Align2, Color32, RichText, Ui};

use crate::hud::HotbarSlot;
use crate::settings::{ControlAction, Settings};
use crate::settings_menu::Tab;
use crate::style::{BACKGROUND, TEXT};
use crate::widgets::{self, BUTTON_SIZE, Images, PANEL_MARGIN, stone_button};
use crate::{I18n, Language};

/// Seconds the startup screen takes to give way to the menu.
const SPLASH_FADE: f32 = 0.8;
/// Seconds the title picture takes to give way to the world.
const PICTURE_FADE: f32 = 1.2;

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
    /// While the world around the player loads.
    Loading,
    /// At startup, while the world behind the title menu loads.
    Splash,
    /// Over a running game: every block, to put in the hotbar.
    Inventory,
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
    /// Put a block of the catalog in a hotbar cell, or empty the cell.
    SetHotbar {
        slot: usize,
        block: Option<usize>,
    },
    /// Swap two hotbar cells.
    SwapHotbar(usize, usize),
}

/// The blocks the inventory shows and where they go.
#[derive(Clone, Copy, Debug)]
pub struct InventoryContext<'a> {
    /// Every block that can be put in the hotbar.
    pub catalog: &'a [HotbarSlot],
    /// What each cell of the hotbar holds; `None` for an empty one.
    pub hotbar: &'a [Option<HotbarSlot>],
    /// The hotbar cell in hand.
    pub selected: usize,
}

/// Things the menu shows but does not own.
#[derive(Clone, Copy)]
pub struct MenuContext<'a> {
    pub i18n: &'a I18n,
    pub images: &'a Images,
    pub inventory: InventoryContext<'a>,
    /// The largest interface size that fits the window, in percent.
    pub max_scale: u8,
    /// The world is drawn behind the menu, so the title picture gives way
    /// to it.
    pub world_visible: bool,
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
    /// The page of the settings that is open.
    pub(crate) tab: Tab,
    /// The action whose button the player is picking: the next one pressed
    /// is it.
    pub(crate) listening: Option<ControlAction>,
    /// The block of the catalog held on the pointer in the inventory.
    pub(crate) carried: Option<usize>,
}

impl Menu {
    pub fn new(screen: Screen) -> Self {
        Self {
            screen,
            tab: Tab::default(),
            listening: None,
            carried: None,
        }
    }

    /// Whether the player is picking a button for an action: the next
    /// button pressed is for [`Menu::offer_button`].
    pub fn is_listening(&self) -> bool {
        self.listening.is_some()
    }

    /// Binds the button named `button` (see the input's names) to the action
    /// being picked for, if there is one.
    pub fn offer_button(&mut self, settings: &mut Settings, button: &str) {
        if let Some(action) = self.listening.take() {
            settings.controls.set_key(action, button);
        }
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
            Screen::Settings { .. } => self.settings_screen(ui, context, settings),
            Screen::Loading => {
                // Under another name, so leaving the loading screen does not
                // show the title menu the startup screen fading away.
                let caption = context.i18n.get("menu-loading");
                splash(ui, context.images, "loading", true, Some(&caption));
                None
            }
            Screen::Splash => {
                splash(ui, context.images, "splash", true, None);
                None
            }
            Screen::Inventory => self.inventory_screen(ui, context),
        }
    }

    /// What Escape does: leaves the settings or resumes the game.
    pub fn back(&mut self) -> Option<MenuAction> {
        // Escape gives up picking a button before it leaves the settings.
        if self.listening.take().is_some() {
            return None;
        }
        match self.screen {
            Screen::Main | Screen::Loading | Screen::Splash => None,
            Screen::Paused | Screen::Inventory => Some(MenuAction::Resume),
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
        let images = context.images;
        let mut action = None;
        background(ui, images, context.world_visible);
        centered(ui, "main menu", |ui| {
            if let Some(logo) = &images.logo {
                let width = (ui.ctx().content_rect().width() * 0.42).clamp(360.0, 720.0);
                ui.add(egui::Image::new(logo).max_width(width));
                ui.add_space(20.0);
            }
            ui.spacing_mut().item_spacing.y = 16.0;
            if stone_button(ui, images, &t("menu-singleplayer"), true).clicked() {
                action = Some(MenuAction::StartSingleplayer);
            }
            stone_button(ui, images, &t("menu-multiplayer"), false)
                .on_hover_text(t("menu-coming-soon"));
            if stone_button(ui, images, &t("menu-settings"), true).clicked() {
                self.screen = Screen::Settings { from_game: false };
            }
            if stone_button(ui, images, &t("menu-exit"), true).clicked() {
                action = Some(MenuAction::Exit);
            }
        });
        egui::Area::new("version".into())
            .anchor(Align2::LEFT_BOTTOM, [12.0, -10.0])
            .show(ui.ctx(), |ui| {
                ui.label(
                    RichText::new(format!("Ruda {}", context.version))
                        .small()
                        .color(Color32::from_white_alpha(140)),
                );
            });
        // The startup screen leaves over the menu instead of cutting to it.
        splash(ui, images, "splash", false, None);
        action
    }

    fn paused(&mut self, ui: &mut Ui, context: MenuContext<'_>) -> Option<MenuAction> {
        let (images, i18n) = (context.images, context.i18n);
        dim(ui);
        let mut action = None;
        egui::Area::new("pause menu".into())
            .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
            .show(ui.ctx(), |ui| {
                // The title, then the three buttons.
                let height = 56.0 + 3.0 * BUTTON_SIZE.y + 3.0 * 16.0;
                let size = egui::vec2(
                    BUTTON_SIZE.x + 2.0 * PANEL_MARGIN,
                    height + 2.0 * PANEL_MARGIN,
                );
                let (rect, _) = ui.allocate_exact_size(size, egui::Sense::hover());
                let rect = rect.round_to_pixels(ui.pixels_per_point());
                widgets::panel(ui.painter(), images, rect);
                let inner = rect.shrink(PANEL_MARGIN);
                widgets::heading(
                    ui.painter(),
                    egui::pos2(inner.center().x, inner.min.y + 24.0),
                    &i18n.get("menu-paused"),
                );
                let buttons = egui::Rect::from_min_max(
                    egui::pos2(inner.min.x, inner.min.y + 56.0 + 16.0),
                    inner.max,
                );
                ui.scope_builder(egui::UiBuilder::new().max_rect(buttons), |ui| {
                    ui.spacing_mut().item_spacing.y = 16.0;
                    if stone_button(ui, images, &i18n.get("menu-resume"), true).clicked() {
                        action = Some(MenuAction::Resume);
                    }
                    if stone_button(ui, images, &i18n.get("menu-settings"), true).clicked() {
                        self.screen = Screen::Settings { from_game: true };
                    }
                    if stone_button(ui, images, &i18n.get("menu-quit-to-title"), true).clicked() {
                        action = Some(MenuAction::QuitToTitle);
                    }
                });
            });
        action
    }
}

/// What the main menu stands in front of: the live world, the picture of it
/// until the world is there, or the plain colour without either.
pub(crate) fn background(ui: &mut Ui, images: &Images, world_visible: bool) {
    // The picture fades out once the world takes its place, and back in
    // when the world goes. After being away, say in a game, it starts afresh
    // rather than carrying on from where the last visit left it.
    let ctx = ui.ctx();
    let seen = egui::Id::new("title picture seen");
    let pass = ctx.cumulative_pass_nr();
    let (visit, last): (u64, u64) = ctx.data(|data| data.get_temp(seen)).unwrap_or_default();
    let visit = if pass > last + 2 { visit + 1 } else { visit };
    ctx.data_mut(|data| data.insert_temp(seen, (visit, pass)));
    let picture = ctx.animate_bool_with_time(
        egui::Id::new(("title picture", visit)),
        !world_visible,
        PICTURE_FADE,
    );
    let picture = images.background.as_ref().map(|texture| (texture, picture));
    if picture.is_some() || world_visible {
        widgets::backdrop(ui, picture, ui.ctx().content_rect());
    }
}

/// The startup screen, over everything: the logo and a bar to wait by. Once
/// `shown` is false it fades away, and then it is not drawn at all.
fn splash(ui: &Ui, images: &Images, id: &'static str, shown: bool, caption: Option<&str>) {
    let opacity = ui
        .ctx()
        .animate_bool_with_time(egui::Id::new(id), shown, SPLASH_FADE);
    if opacity <= 0.0 {
        return;
    }
    let screen = ui.ctx().content_rect();
    let painter = ui.ctx().layer_painter(egui::LayerId::new(
        egui::Order::Foreground,
        egui::Id::new(id),
    ));
    painter.rect_filled(screen, 0.0, BACKGROUND.gamma_multiply(opacity));
    let mut bottom = screen.center().y;
    if let Some(logo) = &images.logo {
        let width = (screen.width() * 0.42).clamp(360.0, 720.0);
        let size = logo.size_vec2() * (width / logo.size_vec2().x);
        let center = egui::pos2(screen.center().x, screen.center().y - 30.0);
        let rect = egui::Rect::from_center_size(center, size);
        let uv = egui::Rect::from_min_max(egui::Pos2::ZERO, egui::pos2(1.0, 1.0));
        painter.image(logo.id(), rect, uv, Color32::WHITE.gamma_multiply(opacity));
        bottom = rect.max.y;
    }
    let middle = screen.center().x;
    let mut bar = bottom + 48.0;
    if let Some(caption) = caption {
        let at = egui::pos2(middle, bottom + 36.0);
        widgets::pixel_text(&painter, at, caption, 24.0, TEXT, true, opacity);
        bar += 40.0;
    }
    if shown {
        let time = ui.input(|input| input.time);
        widgets::loading_bar(&painter, egui::pos2(middle, bar), time);
    }
}

/// Lays out `contents` in a column centred on the screen.
pub(crate) fn centered(ui: &mut Ui, id: &str, contents: impl FnOnce(&mut Ui)) {
    egui::Area::new(egui::Id::new(id))
        .anchor(Align2::CENTER_CENTER, [0.0, 0.0])
        .show(ui.ctx(), |ui| {
            ui.vertical_centered(contents);
        });
}

/// Darkens the game behind a menu.
pub(crate) fn dim(ui: &mut Ui) {
    let screen = ui.ctx().content_rect();
    ui.painter()
        .rect_filled(screen, 0.0, Color32::from_black_alpha(150));
}
