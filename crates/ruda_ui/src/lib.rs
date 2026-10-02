//! Menus and settings, drawn with egui, and their translations.

mod i18n;
mod menu;
mod settings;
mod style;

pub use i18n::{I18n, Language};
pub use menu::{Menu, MenuAction, MenuContext, Screen};
pub use settings::{
    Controls, FIELDS_OF_VIEW, FpsLimit, GpuApi, Graphics, LOD_DISTANCES, Lighting, Preset,
    Settings, Shadows, VIEW_DISTANCES,
};
pub use style::{BACKGROUND, apply as apply_style};
