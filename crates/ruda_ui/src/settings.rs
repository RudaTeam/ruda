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
    pub controls: Controls,
}

/// How the game handles.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Controls {
    /// Jump onto a block the player walks into without pressing jump.
    pub auto_jump: bool,
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
    /// The view sways with the player's steps.
    pub view_bobbing: bool,
    pub fps_limit: FpsLimit,
    pub fullscreen: bool,
    /// Shadows cast by the sun and moon; costly.
    #[serde(deserialize_with = "shadows_cast")]
    pub shadows: Shadows,
    /// Clouds over the world.
    #[serde(deserialize_with = "clouds_shown")]
    pub clouds: bool,
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
            view_bobbing: true,
            fps_limit: FpsLimit::default(),
            fullscreen: false,
            shadows: Shadows::Off,
            clouds: true,
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
            Preset::Standard => (Lighting::Classic, true, Shadows::Off, 8, 512),
            Preset::High => (Lighting::Atmospheric, true, Shadows::Off, 12, 1024),
            Preset::Ultra => (Lighting::Atmospheric, true, Shadows::Rays, 16, 2048),
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

/// How the sun and the moon cast shadows.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Shadows {
    #[default]
    Off,
    /// From a map of what the sun sees.
    Standard,
    /// Traced towards the sun for every point near the camera: exact.
    Rays,
}

impl Shadows {
    pub const ALL: [Shadows; 3] = [Shadows::Off, Shadows::Standard, Shadows::Rays];
}

/// How shadows are cast. Older settings only said whether there were any:
/// those that had them get the best.
fn shadows_cast<'de, D: Deserializer<'de>>(deserializer: D) -> Result<Shadows, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Saved {
        On(bool),
        Kind(Shadows),
    }
    Ok(match Saved::deserialize(deserializer)? {
        Saved::On(true) => Shadows::Rays,
        Saved::On(false) => Shadows::Off,
        Saved::Kind(kind) => kind,
    })
}

/// Whether clouds are shown. Older settings named kinds of clouds: any but
/// "off" shows them.
fn clouds_shown<'de, D: Deserializer<'de>>(deserializer: D) -> Result<bool, D::Error> {
    #[derive(Deserialize)]
    #[serde(untagged)]
    enum Saved {
        Shown(bool),
        Named(String),
    }
    Ok(match Saved::deserialize(deserializer)? {
        Saved::Shown(shown) => shown,
        Saved::Named(name) => name != "off",
    })
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
        settings.graphics.clouds = false;
        settings.controls.auto_jump = true;
        assert_eq!(Settings::from_toml(&settings.to_toml()).unwrap(), settings);
    }

    #[test]
    fn fills_in_and_clamps_old_files() {
        let settings = Settings::from_toml("[graphics]\nview_distance = 99\n").unwrap();
        assert_eq!(settings.graphics.view_distance, 16);
        assert_eq!(settings.graphics.fov, 70);
        assert_eq!(settings.language, None);
        assert!(!settings.controls.auto_jump);
        assert!(settings.graphics.view_bobbing);
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
        assert!(graphics.clouds);
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
    fn reads_shadows_from_when_they_were_on_or_off() {
        let shadows = |text| Settings::from_toml(text).unwrap().graphics.shadows;
        assert_eq!(shadows("[graphics]\nshadows = false\n"), Shadows::Off);
        assert_eq!(shadows("[graphics]\nshadows = true\n"), Shadows::Rays);
        assert_eq!(
            shadows("[graphics]\nshadows = \"standard\"\n"),
            Shadows::Standard
        );
        assert_eq!(shadows("[graphics]\n"), Shadows::Off);
    }

    #[test]
    fn reads_clouds_from_when_they_came_in_kinds() {
        let clouds = |text| Settings::from_toml(text).unwrap().graphics.clouds;
        assert!(!clouds("[graphics]\nclouds = false\n"));
        assert!(clouds("[graphics]\nclouds = true\n"));
        assert!(!clouds("[graphics]\nclouds = \"off\"\n"));
        assert!(clouds("[graphics]\nclouds = \"standard\"\n"));
        assert!(clouds("[graphics]\nclouds = \"volumetric\"\n"));
    }
}
