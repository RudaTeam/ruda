//! Menus and settings, drawn with egui, and their translations.

mod controls;
mod hud;
mod i18n;
mod inventory;
mod menu;
mod settings;
mod settings_menu;
mod style;
mod widgets;

pub use hud::{HotbarSlot, Hud, HudContext};
pub use i18n::{I18n, Language};
pub use menu::{InventoryContext, Menu, MenuAction, MenuContext, Screen};
pub use settings::{
    Appearance, ControlAction, Controls, FIELDS_OF_VIEW, FpsLimit, GpuApi, Graphics, LOD_DISTANCES,
    Lighting, MOUSE_SENSITIVITIES, Preset, Settings, Shadows, UI_SCALE_STEP, UI_SCALES,
    VIEW_DISTANCES,
};
pub use style::{BACKGROUND, apply as apply_style};
pub use widgets::{Images, PICTURE_TEXTURE, TILING_TEXTURE};
