//! The settings file.

use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use anyhow::{Context as _, Result};
use ruda_ui::Settings;
use tracing::warn;

/// `~/.config/ruda` on Linux, `~/Library/Application Support/Ruda` on macOS
/// and `%APPDATA%\Ruda` on Windows.
pub fn path() -> Option<PathBuf> {
    let env = |name| std::env::var_os(name).filter(|value| !value.is_empty());
    let dir = if cfg!(windows) {
        PathBuf::from(env("APPDATA")?).join("Ruda")
    } else if cfg!(target_os = "macos") {
        PathBuf::from(env("HOME")?).join("Library/Application Support/Ruda")
    } else {
        env("XDG_CONFIG_HOME")
            .map(PathBuf::from)
            .or_else(|| Some(PathBuf::from(env("HOME")?).join(".config")))?
            .join("ruda")
    };
    Some(dir.join("settings.toml"))
}

/// The saved settings; the defaults if there are none or they can't be read.
pub fn load(path: &Path) -> Settings {
    let text = match std::fs::read_to_string(path) {
        Ok(text) => text,
        Err(error) if error.kind() == ErrorKind::NotFound => return Settings::default(),
        Err(error) => {
            warn!(path = %path.display(), "can't read the settings: {error}");
            return Settings::default();
        }
    };
    Settings::from_toml(&text).unwrap_or_else(|error| {
        warn!(path = %path.display(), "ignoring broken settings: {error}");
        Settings::default()
    })
}

pub fn save(path: &Path, settings: &Settings) -> Result<()> {
    if let Some(dir) = path.parent() {
        std::fs::create_dir_all(dir).with_context(|| format!("can't create {}", dir.display()))?;
    }
    std::fs::write(path, settings.to_toml())
        .with_context(|| format!("can't write {}", path.display()))
}
