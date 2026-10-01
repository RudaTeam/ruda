use std::ops::RangeInclusive;

use serde::{Deserialize, Serialize};

use crate::Language;

/// View distances offered in the settings, in chunks of 32 blocks.
pub const VIEW_DISTANCES: RangeInclusive<u8> = 2..=16;
/// How far the far-away look of the world can reach, in blocks; 0 is off.
pub const LOD_DISTANCES: [u16; 4] = [0, 512, 1024, 2048];
/// Vertical fields of view offered in the settings, in degrees.
pub const FIELDS_OF_VIEW: RangeInclusive<u8> = 50..=110;

/// The player's preferences, kept between runs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// `None` follows the system language.
    pub language: Option<Language>,
    pub graphics: Graphics,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Graphics {
    /// In chunks of 32 blocks.
    pub view_distance: u8,
    /// How far simplified far-away terrain is drawn past the chunks, in
    /// blocks; 0 for none.
    pub lod_distance: u16,
    /// Vertical field of view in degrees.
    pub fov: u8,
    pub vsync: bool,
    pub fullscreen: bool,
    /// Shadows cast by the sun and moon; costly, so off by default.
    pub shadows: bool,
    /// Takes effect on the next start.
    pub gpu_api: GpuApi,
}

impl Default for Graphics {
    fn default() -> Self {
        Self {
            view_distance: 6,
            lod_distance: 1024,
            fov: 70,
            vsync: true,
            fullscreen: false,
            shadows: false,
            gpu_api: GpuApi::Auto,
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum GpuApi {
    #[default]
    Auto,
    Vulkan,
    Metal,
    Dx12,
    Gl,
}

impl GpuApi {
    /// The choices that can work on this platform.
    pub fn available() -> &'static [GpuApi] {
        if cfg!(target_os = "macos") || cfg!(target_os = "ios") {
            &[GpuApi::Auto, GpuApi::Metal]
        } else if cfg!(windows) {
            &[GpuApi::Auto, GpuApi::Dx12, GpuApi::Vulkan, GpuApi::Gl]
        } else {
            &[GpuApi::Auto, GpuApi::Vulkan, GpuApi::Gl]
        }
    }

    /// Name for the picker; `None` for [`GpuApi::Auto`], which is translated.
    pub fn name(self) -> Option<&'static str> {
        match self {
            GpuApi::Auto => None,
            GpuApi::Vulkan => Some("Vulkan"),
            GpuApi::Metal => Some("Metal"),
            GpuApi::Dx12 => Some("DirectX 12"),
            GpuApi::Gl => Some("OpenGL"),
        }
    }
}

impl Settings {
    /// Reads settings saved by [`Settings::to_toml`]. Missing values take
    /// their defaults and out-of-range ones are clamped, so old or
    /// hand-edited files still load.
    pub fn from_toml(text: &str) -> Result<Self, toml::de::Error> {
        let mut settings: Self = toml::from_str(text)?;
        let graphics = &mut settings.graphics;
        graphics.view_distance = graphics
            .view_distance
            .clamp(*VIEW_DISTANCES.start(), *VIEW_DISTANCES.end());
        graphics.fov = graphics
            .fov
            .clamp(*FIELDS_OF_VIEW.start(), *FIELDS_OF_VIEW.end());
        // The nearest of the offered distances.
        graphics.lod_distance = LOD_DISTANCES
            .into_iter()
            .min_by_key(|&distance| distance.abs_diff(graphics.lod_distance))
            .expect("there are distances");
        Ok(settings)
    }

    pub fn to_toml(&self) -> String {
        toml::to_string_pretty(self).expect("settings always serialize")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trips_through_toml() {
        let mut settings = Settings {
            language: Some(Language::Russian),
            ..Default::default()
        };
        settings.graphics.vsync = false;
        settings.graphics.gpu_api = GpuApi::Gl;
        assert_eq!(Settings::from_toml(&settings.to_toml()).unwrap(), settings);
    }

    #[test]
    fn fills_in_and_clamps_old_files() {
        let settings = Settings::from_toml("[graphics]\nview_distance = 99\n").unwrap();
        assert_eq!(settings.graphics.view_distance, 16);
        assert_eq!(settings.graphics.fov, 70);
        assert_eq!(settings.language, None);
    }
}
