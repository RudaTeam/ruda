//! Menus and settings, drawn with egui, and their translations.

mod i18n;
mod menu;
mod settings;
mod style;

pub use i18n::{I18n, Language};
pub use menu::{Menu, MenuAction, MenuContext, Screen};
pub use settings::{FIELDS_OF_VIEW, GpuApi, Graphics, LOD_DISTANCES, Settings, VIEW_DISTANCES};
pub use style::{BACKGROUND, apply as apply_style};
