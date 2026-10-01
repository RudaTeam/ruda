/// The light at a block: light from the sky and the red, green and blue
/// light that blocks give off, each from 0 (dark) to 15, packed into 16 bits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Serialize, serde::Deserialize),
    serde(transparent)
)]
pub struct Light(u16);

impl Light {
    /// The brightest level of a channel.
    pub const MAX: u8 = 15;
    /// Sky, red, green and blue.
    pub const CHANNELS: usize = 4;
    pub const SKY_CHANNEL: usize = 0;
    pub const DARK: Self = Self(0);
    /// Under the open sky, with no light from blocks.
    pub const SKY: Self = Self(Self::MAX as u16);

    /// Levels above [`Light::MAX`] are cut down to it.
    pub const fn new(sky: u8, red: u8, green: u8, blue: u8) -> Self {
        Self::DARK
            .with_channel(0, sky)
            .with_channel(1, red)
            .with_channel(2, green)
            .with_channel(3, blue)
    }

    /// Light of a block, without sky light.
    pub const fn rgb(red: u8, green: u8, blue: u8) -> Self {
        Self::new(0, red, green, blue)
    }

    pub const fn sky(self) -> u8 {
        self.channel(Self::SKY_CHANNEL)
    }

    pub const fn red(self) -> u8 {
        self.channel(1)
    }

    pub const fn green(self) -> u8 {
        self.channel(2)
    }

    pub const fn blue(self) -> u8 {
        self.channel(3)
    }

    /// Channel 0 is the sky, 1 to 3 red, green and blue.
    pub const fn channel(self, channel: usize) -> u8 {
        ((self.0 >> (4 * channel)) & 0xf) as u8
    }

    pub const fn with_channel(self, channel: usize, level: u8) -> Self {
        let shift = 4 * channel;
        let level = if level > Self::MAX { Self::MAX } else { level };
        Self((self.0 & !(0xf << shift)) | (level as u16) << shift)
    }

    /// The brighter of the two in every channel.
    pub const fn max(self, other: Self) -> Self {
        let mut light = self;
        let mut channel = 0;
        while channel < Self::CHANNELS {
            if other.channel(channel) > light.channel(channel) {
                light = light.with_channel(channel, other.channel(channel));
            }
            channel += 1;
        }
        light
    }

    /// Without sky light: what blocks contribute.
    pub const fn without_sky(self) -> Self {
        Self(self.0 & !0xf)
    }

    pub const fn is_dark(self) -> bool {
        self.0 == 0
    }

    pub const fn to_raw(self) -> u16 {
        self.0
    }

    pub const fn from_raw(raw: u16) -> Self {
        Self(raw)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn packs_four_channels() {
        let light = Light::new(15, 1, 7, 20);
        assert_eq!(
            (light.sky(), light.red(), light.green(), light.blue()),
            (15, 1, 7, 15)
        );
        assert_eq!(light.with_channel(2, 3).green(), 3);
        assert_eq!(
            Light::new(2, 9, 0, 4).max(Light::new(5, 1, 8, 4)),
            Light::new(5, 9, 8, 4)
        );
        assert_eq!(Light::new(12, 1, 2, 3).without_sky(), Light::rgb(1, 2, 3));
        assert_eq!(Light::SKY.sky(), 15);
    }
}
