use std::collections::BTreeMap;
use std::ops::RangeInclusive;

use serde::{Deserialize, Deserializer, Serialize};

use crate::Language;

/// View distances offered in the settings, in chunks of 32 blocks.
pub const VIEW_DISTANCES: RangeInclusive<u8> = 2..=16;
/// How far the far-away look of the world can reach, in blocks; 0 is off.
pub const LOD_DISTANCES: [u16; 4] = [0, 512, 1024, 2048];
/// Vertical fields of view offered in the settings, in degrees.
pub const FIELDS_OF_VIEW: RangeInclusive<u8> = 50..=110;
/// How fast the mouse turns the camera, in percent of the normal speed.
pub const MOUSE_SENSITIVITIES: RangeInclusive<u8> = 20..=200;
/// How large the interface is drawn, in percent, in steps of
/// [`UI_SCALE_STEP`].
pub const UI_SCALES: RangeInclusive<u8> = 75..=200;
pub const UI_SCALE_STEP: u8 = 25;

/// The player's preferences, kept between runs.
#[derive(Clone, Debug, Default, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct Settings {
    /// `None` follows the system language.
    pub language: Option<Language>,
    pub graphics: Graphics,
    pub controls: Controls,
    pub appearance: Appearance,
}

/// How the menus look.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Appearance {
    /// How large the interface is drawn, in percent.
    pub scale: u8,
}

impl Default for Appearance {
    fn default() -> Self {
        Self { scale: 100 }
    }
}

/// How the game handles.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default)]
pub struct Controls {
    /// Jump onto a block the player walks into without pressing jump.
    pub auto_jump: bool,
    /// How fast the mouse turns the camera, in percent of the normal speed.
    pub mouse_sensitivity: u8,
    /// The buttons picked for actions, by [`ControlAction::id`]; the actions
    /// left out keep their usual ones.
    keys: BTreeMap<String, String>,
}

impl Default for Controls {
    fn default() -> Self {
        Self {
            auto_jump: false,
            mouse_sensitivity: 100,
            keys: BTreeMap::new(),
        }
    }
}

impl Controls {
    /// The name of the button `action` is bound to.
    pub fn key(&self, action: ControlAction) -> &str {
        self.keys
            .get(action.id())
            .map_or(action.default_key(), String::as_str)
    }

    /// Binds `button` to `action`. Another action that had it takes the
    /// button `action` had, so no two share one.
    pub fn set_key(&mut self, action: ControlAction, button: &str) {
        let before = self.key(action).to_owned();
        for other in ControlAction::ALL {
            if other != action && self.key(other) == button {
                self.keys.insert(other.id().to_owned(), before.clone());
            }
        }
        self.keys.insert(action.id().to_owned(), button.to_owned());
    }

    /// Gives every action its usual button back.
    pub fn reset_keys(&mut self) {
        self.keys.clear();
    }

    /// Keeps only the picked buttons `usable` accepts: the rest, such as a
    /// name from a file edited by hand, go back to the usual ones. If two
    /// actions end up on one button, which no one can have picked, all go
    /// back.
    pub fn retain_keys(&mut self, usable: impl Fn(&str) -> bool) {
        self.keys.retain(|_, button| usable(button));
        let mut seen = std::collections::HashSet::new();
        if !self.keys().all(|(_, button)| seen.insert(button)) {
            self.keys.clear();
        }
    }

    /// The buttons of all actions, for those that have changed.
    pub fn keys(&self) -> impl Iterator<Item = (ControlAction, &str)> {
        ControlAction::ALL
            .into_iter()
            .map(|action| (action, self.key(action)))
    }
}

/// What a player can choose the button for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ControlAction {
    MoveForward,
    MoveBack,
    MoveLeft,
    MoveRight,
    Jump,
    Sneak,
    Sprint,
    Break,
    Place,
    Inventory,
}

impl ControlAction {
    pub const ALL: [ControlAction; 10] = [
        ControlAction::MoveForward,
        ControlAction::MoveBack,
        ControlAction::MoveLeft,
        ControlAction::MoveRight,
        ControlAction::Jump,
        ControlAction::Sneak,
        ControlAction::Sprint,
        ControlAction::Break,
        ControlAction::Place,
        ControlAction::Inventory,
    ];

    /// What the action is called in the settings file.
    pub fn id(self) -> &'static str {
        match self {
            ControlAction::MoveForward => "move_forward",
            ControlAction::MoveBack => "move_back",
            ControlAction::MoveLeft => "move_left",
            ControlAction::MoveRight => "move_right",
            ControlAction::Jump => "jump",
            ControlAction::Sneak => "sneak",
            ControlAction::Sprint => "sprint",
            ControlAction::Break => "break",
            ControlAction::Place => "place",
            ControlAction::Inventory => "inventory",
        }
    }

    /// The button it is bound to until the player picks another, as a name
    /// the input understands.
    pub fn default_key(self) -> &'static str {
        match self {
            ControlAction::MoveForward => "KeyW",
            ControlAction::MoveBack => "KeyS",
            ControlAction::MoveLeft => "KeyA",
            ControlAction::MoveRight => "KeyD",
            ControlAction::Jump => "Space",
            ControlAction::Sneak => "ShiftLeft",
            ControlAction::Sprint => "ControlLeft",
            ControlAction::Break => "mouse:Left",
            ControlAction::Place => "mouse:Right",
            ControlAction::Inventory => "KeyE",
        }
    }
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
    /// A world drifts behind the title menu instead of a picture of it.
    pub menu_world: bool,
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
            menu_world: true,
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
        let controls = &mut settings.controls;
        controls.mouse_sensitivity = controls
            .mouse_sensitivity
            .clamp(*MOUSE_SENSITIVITIES.start(), *MOUSE_SENSITIVITIES.end());
        // Only the actions there are; the rest of what a newer or edited
        // file says is forgotten.
        controls
            .keys
            .retain(|id, _| ControlAction::ALL.iter().any(|action| action.id() == id));
        settings.controls.retain_keys(|_| true);
        let scale = &mut settings.appearance.scale;
        // The nearest of the offered scales.
        *scale = UI_SCALES
            .step_by(usize::from(UI_SCALE_STEP))
            .min_by_key(|&offered| offered.abs_diff(*scale))
            .expect("there are scales");
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
        settings.graphics.menu_world = false;
        settings.controls.auto_jump = true;
        settings.controls.mouse_sensitivity = 140;
        settings.controls.set_key(ControlAction::Jump, "KeyF");
        settings.appearance.scale = 150;
        assert_eq!(Settings::from_toml(&settings.to_toml()).unwrap(), settings);
    }

    #[test]
    fn buttons_default_until_changed_and_never_repeat() {
        let mut controls = Controls::default();
        assert_eq!(controls.key(ControlAction::Jump), "Space");
        controls.set_key(ControlAction::Jump, "KeyF");
        assert_eq!(controls.key(ControlAction::Jump), "KeyF");
        // Taking a button swaps it with its owner.
        controls.set_key(ControlAction::Sprint, "KeyF");
        assert_eq!(controls.key(ControlAction::Sprint), "KeyF");
        assert_eq!(controls.key(ControlAction::Jump), "ControlLeft");
        let buttons: Vec<_> = controls.keys().map(|(_, button)| button).collect();
        for button in &buttons {
            assert_eq!(buttons.iter().filter(|other| *other == button).count(), 1);
        }
        controls.reset_keys();
        assert_eq!(controls.key(ControlAction::Sprint), "ControlLeft");
    }

    #[test]
    fn picked_buttons_that_clash_or_are_unusable_are_dropped() {
        let from = |text| Settings::from_toml(text).unwrap().controls;
        // Two actions on one button: back to the usual ones.
        let clash = from("[controls.keys]\njump = \"KeyW\"\n");
        assert_eq!(clash.key(ControlAction::Jump), "Space");
        assert_eq!(clash.key(ControlAction::MoveForward), "KeyW");
        // A name the input does not know is let go by the caller.
        let mut controls = from("[controls.keys]\njump = \"Garbage\"\nsneak = \"KeyQ\"\n");
        controls.retain_keys(|name| name != "Garbage");
        assert_eq!(controls.key(ControlAction::Jump), "Space");
        assert_eq!(controls.key(ControlAction::Sneak), "KeyQ");
        // A swap that the player made is a fine file.
        let mut swapped = Controls::default();
        swapped.set_key(ControlAction::Jump, "KeyW");
        let swapped = from(
            &Settings {
                controls: swapped,
                ..Default::default()
            }
            .to_toml(),
        );
        assert_eq!(swapped.key(ControlAction::Jump), "KeyW");
        assert_eq!(swapped.key(ControlAction::MoveForward), "Space");
    }

    #[test]
    fn reads_the_new_settings_from_odd_files() {
        let settings = Settings::from_toml(
            "[controls]\nmouse_sensitivity = 250\n[controls.keys]\njump = \"KeyF\"\nfly = \"KeyG\"\n\
             [appearance]\nscale = 130\n",
        )
        .unwrap();
        assert_eq!(settings.controls.mouse_sensitivity, 200);
        assert_eq!(settings.controls.key(ControlAction::Jump), "KeyF");
        assert_eq!(settings.controls.keys.len(), 1);
        assert_eq!(settings.appearance.scale, 125);
        let settings = Settings::from_toml("").unwrap();
        assert_eq!(settings.controls.mouse_sensitivity, 100);
        assert_eq!(settings.appearance.scale, 100);
    }

    #[test]
    fn fills_in_and_clamps_old_files() {
        let settings = Settings::from_toml("[graphics]\nview_distance = 99\n").unwrap();
        assert_eq!(settings.graphics.view_distance, 16);
        assert_eq!(settings.graphics.fov, 70);
        assert_eq!(settings.language, None);
        assert!(!settings.controls.auto_jump);
        assert!(settings.graphics.view_bobbing);
        assert!(settings.graphics.menu_world);
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
