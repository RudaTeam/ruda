use egui::emath::GuiRounding;
use egui::{Align2, Color32, RichText, Ui};

use crate::hud::HotbarSlot;
use crate::settings::{ControlAction, Settings};
use crate::settings_menu::Tab;
use crate::style::{BACKGROUND, TEXT};
use crate::widgets::{self, BUTTON_GAP, BUTTON_SIZE, Images, PANEL_MARGIN, stone_button};
use crate::{I18n, Language};

/// Between the buttons of the main menu.
const MAIN_GAP: f32 = BUTTON_GAP;
/// The settings button of the main menu, beside the language button: it
/// takes what the language button and the gap leave of a row.
const MAIN_SETTINGS: f32 = BUTTON_SIZE.x - BUTTON_SIZE.y - MAIN_GAP;
/// Rows of buttons on the main menu.
const MAIN_ROWS: f32 = 4.0;

/// Where the parts of the main menu go on a screen.
struct MainLayout {
    logo_width: f32,
    /// From the top of the screen to the logo.
    logo_top: f32,
    /// From the top of the screen to the first button.
    buttons_top: f32,
}

/// Where the logo and the buttons go on `screen`, given the logo's height as
/// a fraction of its width: the logo a tenth of the height from the top, the
/// buttons from about two fifths down, never so low that they leave the
/// screen, and the logo smaller if a short screen leaves it no room.
fn main_layout(screen: egui::Rect, logo_ratio: Option<f32>) -> MainLayout {
    let logo_width = (screen.width() * 0.42).clamp(360.0, 720.0);
    let logo_height = logo_ratio.map_or(0.0, |ratio| ratio * logo_width);
    let logo_top = (screen.height() * 0.10).max(24.0);
    let rows = MAIN_ROWS * BUTTON_SIZE.y + (MAIN_ROWS - 1.0) * MAIN_GAP;
    let lowest = (screen.height() - rows - 32.0).max(0.0);
    let buttons_top = (logo_top + logo_height + screen.height() * 0.06)
        .max(screen.height() * 0.42)
        .min(lowest);
    // The logo gives way before it would reach the buttons.
    let room = (buttons_top - logo_top - 12.0).max(0.0);
    let logo_width = match logo_ratio {
        Some(ratio) if ratio * logo_width > room => room / ratio,
        _ => logo_width,
    };
    MainLayout {
        logo_width,
        logo_top,
        buttons_top,
    }
}

/// Seconds the startup screen takes to give way to the menu.
const SPLASH_FADE: f32 = 0.8;
/// Seconds the sunset takes to give way to a live world.
const SCENE_FADE: f32 = 1.2;

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
    /// Picking the language, from the main menu.
    Language,
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
    /// The world is drawn behind the menu, so the sunset gives way to it.
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
            Screen::Language => self.language_screen(ui, context, settings),
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
            Screen::Language => {
                self.screen = Screen::Main;
                Some(MenuAction::SettingsClosed)
            }
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

        // As in Minecraft, the logo is near the top and the buttons start a
        // little above the middle.
        let ratio = images.logo.as_ref().map(|logo| {
            let size = logo.size_vec2();
            size.y / size.x
        });
        let layout = main_layout(ui.ctx().content_rect(), ratio);
        // On a window too short for it the logo is left out.
        if let Some(logo) = images.logo.as_ref().filter(|_| layout.logo_width >= 64.0) {
            egui::Area::new("main logo".into())
                .anchor(Align2::CENTER_TOP, [0.0, layout.logo_top])
                .interactable(false)
                .show(ui.ctx(), |ui| {
                    ui.add(egui::Image::new(logo).max_width(layout.logo_width));
                });
        }
        egui::Area::new("main buttons".into())
            .anchor(Align2::CENTER_TOP, [0.0, layout.buttons_top])
            .show(ui.ctx(), |ui| {
                ui.spacing_mut().item_spacing = egui::vec2(MAIN_GAP, MAIN_GAP);
                if stone_button(ui, images, &t("menu-singleplayer"), true).clicked() {
                    action = Some(MenuAction::StartSingleplayer);
                }
                stone_button(ui, images, &t("menu-multiplayer"), false)
                    .on_hover_text(t("menu-coming-soon"));
                // The language, a square button, and the settings beside it,
                // together as wide as the buttons above.
                ui.horizontal(|ui| {
                    let label = t("settings-language");
                    let globe = widgets::stone_icon_button(
                        ui,
                        images,
                        egui::vec2(BUTTON_SIZE.y, BUTTON_SIZE.y),
                        &label,
                        |painter, center, _| widgets::paint_globe(painter, center, 3.0),
                    );
                    if globe.clicked() {
                        self.screen = Screen::Language;
                    }
                    let wide = egui::vec2(MAIN_SETTINGS, BUTTON_SIZE.y);
                    if widgets::stone_button_sized(ui, images, &t("menu-settings"), wide, true)
                        .clicked()
                    {
                        self.screen = Screen::Settings { from_game: false };
                    }
                });
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

/// What the main menu stands in front of: the pixel sunset, or the live world
/// when that is on, with the sunset there until the world has loaded.
pub(crate) fn background(ui: &mut Ui, images: &Images, world_visible: bool) {
    // The scene fades out once a live world takes its place, and back in
    // when the world goes. After being away, say in a game, it starts afresh
    // rather than carrying on from where the last visit left it.
    let ctx = ui.ctx();
    let seen = egui::Id::new("title scene seen");
    let pass = ctx.cumulative_pass_nr();
    let (visit, last): (u64, u64) = ctx.data(|data| data.get_temp(seen)).unwrap_or_default();
    let visit = if pass > last + 2 { visit + 1 } else { visit };
    ctx.data_mut(|data| data.insert_temp(seen, (visit, pass)));
    let scene = ctx.animate_bool_with_time(
        egui::Id::new(("title scene", visit)),
        !world_visible,
        SCENE_FADE,
    );
    widgets::backdrop(ui, images, ui.ctx().content_rect(), scene);
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

/// Darkens the game behind a menu.
pub(crate) fn dim(ui: &mut Ui) {
    let screen = ui.ctx().content_rect();
    ui.painter()
        .rect_filled(screen, 0.0, Color32::from_black_alpha(150));
}

#[cfg(test)]
mod tests {
    use egui::{Context, Event, Modifiers, PointerButton, Pos2, RawInput, Rect, pos2, vec2};

    use super::*;

    fn screen(width: f32, height: f32) -> Rect {
        Rect::from_min_size(Pos2::ZERO, vec2(width, height))
    }

    /// The main screen as the pointer sees it: egui frames with events.
    struct Table {
        ctx: Context,
        menu: Menu,
        settings: Settings,
        i18n: I18n,
        images: Images,
        time: f64,
    }

    impl Table {
        const SCREEN: (f32, f32) = (1280.0, 800.0);

        fn new() -> Self {
            let ctx = Context::default();
            crate::style::apply(&ctx);
            Self {
                ctx,
                menu: Menu::new(Screen::Main),
                settings: Settings::default(),
                i18n: I18n::new(crate::Language::English),
                images: Images::default(),
                time: 0.0,
            }
        }

        fn frame(&mut self, events: Vec<Event>) -> Option<MenuAction> {
            self.time += 0.05;
            let input = RawInput {
                screen_rect: Some(screen(Self::SCREEN.0, Self::SCREEN.1)),
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
                system_language: crate::Language::English,
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
            let button = |pressed| Event::PointerButton {
                pos: at,
                button: PointerButton::Primary,
                pressed,
                modifiers: Modifiers::NONE,
            };
            self.frame(vec![button(true)]);
            self.frame(vec![button(false)])
        }
    }

    /// The middle of the `row`th row of buttons, `x_from_left` from the left
    /// of their column: the language and settings buttons share the third.
    fn button(row: usize, x_from_left: f32) -> Pos2 {
        let (width, height) = Table::SCREEN;
        let layout = main_layout(screen(width, height), None);
        pos2(
            width / 2.0 - BUTTON_SIZE.x / 2.0 + x_from_left,
            layout.buttons_top + row as f32 * (BUTTON_SIZE.y + MAIN_GAP) + BUTTON_SIZE.y / 2.0,
        )
    }

    #[test]
    fn the_logo_is_near_the_top_and_the_buttons_below_it_on_the_screen() {
        for (width, height) in [
            (1280.0, 720.0),
            (1710.0, 1069.0),
            (960.0, 600.0),
            (3840.0, 2160.0),
            (1000.0, 1000.0),
        ] {
            let layout = main_layout(screen(width, height), None);
            assert!(layout.logo_top <= height * 0.12, "{width}x{height}");
            assert!(layout.buttons_top > layout.logo_top);
            // From about two fifths down, as in Minecraft.
            assert!(layout.buttons_top >= height * 0.38, "{width}x{height}");
            let rows = MAIN_ROWS * BUTTON_SIZE.y + (MAIN_ROWS - 1.0) * MAIN_GAP;
            assert!(layout.buttons_top + rows <= height, "{width}x{height}");
        }
        // A window too short for it still keeps the buttons on the screen.
        let layout = main_layout(screen(800.0, 330.0), None);
        assert!(layout.buttons_top >= 0.0);
    }

    #[test]
    fn the_logo_never_reaches_the_buttons() {
        // The logo of the game is about three times as wide as it is tall.
        let ratio = 0.3;
        for (width, height) in [
            (1280.0, 720.0),
            (960.0, 600.0),
            (800.0, 500.0),
            (800.0, 400.0),
            (1920.0, 1080.0),
        ] {
            let layout = main_layout(screen(width, height), Some(ratio));
            let bottom = layout.logo_top + ratio * layout.logo_width;
            assert!(bottom <= layout.buttons_top, "{width}x{height}");
        }
        // Where there is room it keeps its size.
        let layout = main_layout(screen(1710.0, 1069.0), Some(ratio));
        assert!((layout.logo_width - 718.2).abs() < 0.01);
    }

    #[test]
    fn a_row_is_the_width_of_the_buttons_above() {
        let row = BUTTON_SIZE.y + MAIN_SETTINGS + MAIN_GAP;
        assert_eq!(row, BUTTON_SIZE.x);
    }

    #[test]
    fn every_button_of_the_main_screen_does_its_thing() {
        let mut table = Table::new();
        table.frame(vec![]);
        // Singleplayer, and the multiplayer that is not there yet.
        assert_eq!(
            table.click(button(0, BUTTON_SIZE.x / 2.0)),
            Some(MenuAction::StartSingleplayer)
        );
        assert_eq!(table.click(button(1, BUTTON_SIZE.x / 2.0)), None);
        assert_eq!(table.menu.screen, Screen::Main);
        // The row of the language and the settings, then the exit by itself.
        let (language, settings) = (
            BUTTON_SIZE.y / 2.0,
            BUTTON_SIZE.y + MAIN_GAP + MAIN_SETTINGS / 2.0,
        );
        assert_eq!(
            table.click(button(3, BUTTON_SIZE.x / 2.0)),
            Some(MenuAction::Exit)
        );
        assert_eq!(table.click(button(2, settings)), None);
        assert_eq!(table.menu.screen, Screen::Settings { from_game: false });
        let mut table = Table::new();
        table.frame(vec![]);
        assert_eq!(table.click(button(2, language)), None);
        assert_eq!(table.menu.screen, Screen::Language);
    }
}
