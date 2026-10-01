use std::ops::RangeInclusive;

use serde::{Deserialize, Deserializer, Serialize};

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
    pub fps_limit: FpsLimit,
    pub fullscreen: bool,
    /// Shadows cast by the sun and moon; costly.
    pub shadows: bool,
    pub clouds: Clouds,
    pub lighting: Lighting,
    /// Takes effect on the next start.
    pub gpu_api: GpuApi,
}

impl Default for Graphics {
    /// The "High" preset.
    fn default() -> Self {
        let mut graphics = Self {
            view_distance: 12,
            lod_distance: 1024,
            fov: 70,
            fps_limit: FpsLimit::default(),
            fullscreen: false,
            shadows: false,
            clouds: Clouds::default(),
            lighting: Lighting::default(),
            gpu_api: GpuApi::Auto,
        };
        Preset::High.apply(&mut graphics);
        graphics
    }
}

/// A set of graphics settings to start from.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Preset {
    /// As classic block games look without shaders.
    Standard,
    High,
    Ultra,
}

impl Preset {
    pub const ALL: [Preset; 3] = [Preset::Standard, Preset::High, Preset::Ultra];

    /// Sets the settings the preset decides; the rest stay as they are.
    pub fn apply(self, graphics: &mut Graphics) {
        let (lighting, clouds, shadows, view_distance, lod_distance) = match self {
            Preset::Standard => (Lighting::Classic, Clouds::Standard, false, 8, 512),
            Preset::High => (Lighting::Atmospheric, Clouds::Standard, false, 12, 1024),
            Preset::Ultra => (Lighting::Atmospheric, Clouds::Volumetric, true, 16, 2048),
        };
        graphics.lighting = lighting;
        graphics.clouds = clouds;
        graphics.shadows = shadows;
        graphics.view_distance = view_distance;
        graphics.lod_distance = lod_distance;
    }

    /// The preset `graphics` matches, if any.
    pub fn of(graphics: &Graphics) -> Option<Preset> {
        Preset::ALL.into_iter().find(|preset| {
            let mut applied = graphics.clone();
            preset.apply(&mut applied);
            applied == *graphics
        })
    }
}

/// How the world is lit.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Lighting {
    /// A fixed shade per side, a sky of picked colours, fog at the edge.
    Classic,
    /// Sunlight and skylight through the air.
    #[default]
    Atmospheric,
}

impl Lighting {
    pub const ALL: [Lighting; 2] = [Lighting::Classic, Lighting::Atmospheric];
}

/// How many frames a second the game draws at most.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum FpsLimit {
    /// As many as it can.
    #[default]
    #[serde(rename = "unlimited")]
    Unlimited,
    #[serde(rename = "60")]
    Sixty,
    #[serde(rename = "120")]
    HundredTwenty,
    /// As many as the display shows: vertical sync.
    #[serde(rename = "display")]
    Display,
}

impl FpsLimit {
    pub const ALL: [FpsLimit; 4] = [
        FpsLimit::Unlimited,
        FpsLimit::Sixty,
        FpsLimit::HundredTwenty,
        FpsLimit::Display,
    ];

    /// Frames a second, for the limits that are a number.
    pub fn fps(self) -> Option<u32> {
        match self {
            FpsLimit::Sixty => Some(60),
            FpsLimit::HundredTwenty => Some(120),
            FpsLimit::Unlimited | FpsLimit::Display => None,
        }
    }
}

/// How clouds are drawn.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Clouds {
    Off,
    /// Blocks of cloud in a thin layer, as in classic block games.
    #[default]
    Standard,
    /// Soft clouds the light passes through: the best looking, for stronger
    /// graphics cards.
    Volumetric,
}

impl Clouds {
    pub const ALL: [Clouds; 3] = [Clouds::Off, Clouds::Standard, Clouds::Volumetric];
}

impl<'de> Deserialize<'de> for Clouds {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        // Older settings only said whether there were clouds, or had other
        // kinds of them.
        #[derive(Deserialize)]
        #[serde(untagged)]
        enum Saved {
            Shown(bool),
            Named(Named),
        }
        #[derive(Deserialize)]
        #[serde(rename_all = "lowercase")]
        enum Named {
            Off,
            Standard,
            Blocky,
            Simple,
            Volumetric,
        }
        Ok(match Saved::deserialize(deserializer)? {
            Saved::Shown(false) | Saved::Named(Named::Off) => Clouds::Off,
            Saved::Shown(true) | Saved::Named(Named::Standard | Named::Blocky | Named::Simple) => {
                Clouds::Standard
            }
            Saved::Named(Named::Volumetric) => Clouds::Volumetric,
        })
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
        settings.graphics.fps_limit = FpsLimit::HundredTwenty;
        settings.graphics.gpu_api = GpuApi::Gl;
        settings.graphics.clouds = Clouds::Off;
        assert_eq!(Settings::from_toml(&settings.to_toml()).unwrap(), settings);
    }

    #[test]
    fn fills_in_and_clamps_old_files() {
        let settings = Settings::from_toml("[graphics]\nview_distance = 99\n").unwrap();
        assert_eq!(settings.graphics.view_distance, 16);
        assert_eq!(settings.graphics.fov, 70);
        assert_eq!(settings.language, None);
    }

    #[test]
    fn presets_are_recognised_until_changed() {
        let mut graphics = Graphics::default();
        assert_eq!(Preset::of(&graphics), Some(Preset::High));
        for preset in Preset::ALL {
            preset.apply(&mut graphics);
            assert_eq!(Preset::of(&graphics), Some(preset));
        }
        Preset::Standard.apply(&mut graphics);
        assert_eq!(graphics.lighting, Lighting::Classic);
        assert_eq!(graphics.clouds, Clouds::Standard);
        graphics.view_distance = 9;
        assert_eq!(Preset::of(&graphics), None);
    }

    #[test]
    fn frames_are_unlimited_unless_chosen() {
        // Settings from before the limit had `vsync`; it no longer counts.
        let settings = Settings::from_toml("[graphics]\nvsync = true\n").unwrap();
        assert_eq!(settings.graphics.fps_limit, FpsLimit::Unlimited);
        let settings = Settings::from_toml("[graphics]\nfps_limit = \"60\"\n").unwrap();
        assert_eq!(settings.graphics.fps_limit.fps(), Some(60));
    }

    #[test]
    fn reads_clouds_from_before_their_quality() {
        let clouds = |text| Settings::from_toml(text).unwrap().graphics.clouds;
        assert_eq!(clouds("[graphics]\nclouds = false\n"), Clouds::Off);
        assert_eq!(clouds("[graphics]\nclouds = true\n"), Clouds::Standard);
        assert_eq!(
            clouds("[graphics]\nclouds = \"simple\"\n"),
            Clouds::Standard
        );
        assert_eq!(
            clouds("[graphics]\nclouds = \"volumetric\"\n"),
            Clouds::Volumetric
        );
    }
}
