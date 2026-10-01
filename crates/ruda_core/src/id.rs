use std::fmt;
use std::str::FromStr;

/// Name of a piece of content, written `namespace:path`, for example
/// `base:stone`. The namespace belongs to the content pack that registers it.
#[derive(Clone, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct ResourceId {
    /// `namespace:path` in a single allocation; `colon` is where it splits.
    full: Box<str>,
    colon: usize,
}

impl ResourceId {
    /// Namespace of the engine's own content, such as `ruda:air`.
    pub const ENGINE: &'static str = "ruda";

    pub fn new(namespace: &str, path: &str) -> Result<Self, InvalidId> {
        let valid_namespace = !namespace.is_empty() && namespace.bytes().all(is_namespace_byte);
        let valid_path =
            !path.is_empty() && path.bytes().all(|b| is_namespace_byte(b) || b == b'/');
        let full = format!("{namespace}:{path}");
        if !valid_namespace || !valid_path {
            return Err(InvalidId(full));
        }
        Ok(Self {
            full: full.into(),
            colon: namespace.len(),
        })
    }

    pub fn namespace(&self) -> &str {
        &self.full[..self.colon]
    }

    pub fn path(&self) -> &str {
        &self.full[self.colon + 1..]
    }

    pub fn as_str(&self) -> &str {
        &self.full
    }
}

fn is_namespace_byte(b: u8) -> bool {
    b.is_ascii_lowercase() || b.is_ascii_digit() || matches!(b, b'_' | b'-' | b'.')
}

impl FromStr for ResourceId {
    type Err = InvalidId;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let (namespace, path) = s.split_once(':').ok_or_else(|| InvalidId(s.to_owned()))?;
        Self::new(namespace, path)
    }
}

impl fmt::Display for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.full)
    }
}

impl fmt::Debug for ResourceId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        fmt::Debug::fmt(&*self.full, f)
    }
}

/// Serialized as the `namespace:path` string; deserializing validates it.
#[cfg(feature = "serde")]
impl serde::Serialize for ResourceId {
    fn serialize<S: serde::Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.as_str())
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for ResourceId {
    fn deserialize<D: serde::Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = <std::borrow::Cow<'de, str>>::deserialize(deserializer)?;
        s.parse().map_err(serde::de::Error::custom)
    }
}

#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[error(
    "invalid resource id `{0}`: expected `namespace:path` made of lowercase letters, \
     digits and `_-.`, with `/` also allowed in the path"
)]
pub struct InvalidId(String);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_namespace_and_path() {
        let id: ResourceId = "base:blocks/stone".parse().unwrap();
        assert_eq!(id.namespace(), "base");
        assert_eq!(id.path(), "blocks/stone");
        assert_eq!(id.to_string(), "base:blocks/stone");
        assert_eq!(id, ResourceId::new("base", "blocks/stone").unwrap());
    }

    #[test]
    fn rejects_malformed_ids() {
        for bad in [
            "stone",
            ":stone",
            "base:",
            "Base:stone",
            "base:Stone",
            "base:st one",
            "base:a:b",
            "ba/se:stone",
        ] {
            assert!(bad.parse::<ResourceId>().is_err(), "{bad}");
        }
    }
}
