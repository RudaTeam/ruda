use std::fmt;

use fluent_bundle::{FluentArgs, FluentBundle, FluentResource, FluentValue};
use serde::{Deserialize, Serialize};
use unic_langid::LanguageIdentifier;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum Language {
    English,
    Russian,
}

impl Language {
    pub const ALL: [Language; 2] = [Language::English, Language::Russian];

    /// The language's name in itself, for the language picker.
    pub fn native_name(self) -> &'static str {
        match self {
            Language::English => "English",
            Language::Russian => "Русский",
        }
    }

    /// The language for a system locale such as `ru-RU`; English for
    /// anything without a translation.
    pub fn from_locale(locale: &str) -> Self {
        if locale.to_ascii_lowercase().starts_with("ru") {
            Language::Russian
        } else {
            Language::English
        }
    }

    fn id(self) -> LanguageIdentifier {
        match self {
            Language::English => "en".parse(),
            Language::Russian => "ru".parse(),
        }
        .expect("valid language tag")
    }

    fn source(self) -> &'static str {
        match self {
            Language::English => include_str!("../locales/en.ftl"),
            Language::Russian => include_str!("../locales/ru.ftl"),
        }
    }
}

/// Translated UI text. Messages missing from a translation fall back to
/// English.
pub struct I18n {
    language: Language,
    bundle: FluentBundle<FluentResource>,
    english: FluentBundle<FluentResource>,
}

impl I18n {
    pub fn new(language: Language) -> Self {
        Self {
            language,
            bundle: bundle(language),
            english: bundle(Language::English),
        }
    }

    pub fn language(&self) -> Language {
        self.language
    }

    pub fn set_language(&mut self, language: Language) {
        if language != self.language {
            *self = Self::new(language);
        }
    }

    /// The message `id`.
    pub fn get(&self, id: &str) -> String {
        self.format(id, None)
    }

    /// The message `id` with numeric arguments.
    pub fn get_with(&self, id: &str, args: &[(&str, i64)]) -> String {
        let mut fluent_args = FluentArgs::new();
        for &(name, value) in args {
            fluent_args.set(name, FluentValue::from(value));
        }
        self.format(id, Some(&fluent_args))
    }

    fn format(&self, id: &str, args: Option<&FluentArgs<'_>>) -> String {
        for bundle in [&self.bundle, &self.english] {
            if let Some(pattern) = bundle.get_message(id).and_then(|message| message.value()) {
                let mut errors = Vec::new();
                return bundle
                    .format_pattern(pattern, args, &mut errors)
                    .into_owned();
            }
        }
        id.to_owned()
    }
}

fn bundle(language: Language) -> FluentBundle<FluentResource> {
    let resource =
        FluentResource::try_new(language.source().to_owned()).expect("translations parse");
    let mut bundle = FluentBundle::new(vec![language.id()]);
    // Unicode isolation marks around arguments would show up as boxes.
    bundle.set_use_isolating(false);
    bundle
        .add_resource(resource)
        .expect("no duplicate messages");
    bundle
}

impl fmt::Debug for I18n {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("I18n")
            .field("language", &self.language)
            .finish_non_exhaustive()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn message_ids(source: &str) -> Vec<&str> {
        source
            .lines()
            .filter(|line| line.starts_with(|c: char| c.is_ascii_lowercase()))
            .filter_map(|line| line.split_once(" =").map(|(id, _)| id))
            .collect()
    }

    #[test]
    fn every_language_translates_every_message() {
        let english = message_ids(Language::English.source());
        assert!(!english.is_empty());
        for language in Language::ALL {
            assert_eq!(message_ids(language.source()), english, "{language:?}");
        }
    }

    #[test]
    fn russian_plurals_agree_with_numbers() {
        let i18n = I18n::new(Language::Russian);
        let value = |chunks: i64| {
            i18n.get_with(
                "settings-view-distance-value",
                &[("chunks", chunks), ("blocks", chunks * 32)],
            )
        };
        assert_eq!(value(1), "1 чанк (32 блока)");
        assert_eq!(value(2), "2 чанка (64 блока)");
        assert_eq!(value(6), "6 чанков (192 блока)");
        assert_eq!(value(16), "16 чанков (512 блоков)");
        assert_eq!(I18n::new(Language::English).get("menu-exit"), "Exit");
    }

    #[test]
    fn picks_the_language_from_the_system_locale() {
        assert_eq!(Language::from_locale("ru-RU"), Language::Russian);
        assert_eq!(Language::from_locale("en_GB"), Language::English);
        assert_eq!(Language::from_locale("de"), Language::English);
    }
}
