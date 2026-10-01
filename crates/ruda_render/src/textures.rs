//! Block textures: decoded from content PNGs into the layers of one texture
//! array, with mipmaps for distant blocks.

use std::collections::HashMap;

use anyhow::{Context as _, Result, ensure};
use ruda_core::{Content, ResourceId};
use tracing::warn;

/// Edge length of a block texture in pixels.
pub const TEXTURE_SIZE: u32 = 16;
/// 16, 8, 4, 2 and 1 pixels.
pub const MIP_LEVELS: u32 = TEXTURE_SIZE.ilog2() + 1;

/// Block textures as texture array layers. Layer 0 is a magenta checkerboard
/// that stands in for missing or broken textures.
#[derive(Clone, Debug)]
pub struct BlockTextures {
    /// RGBA pixels of every layer for each mip level, largest first.
    pub mips: Vec<Vec<u8>>,
    pub layers: u32,
    layer_of: HashMap<ResourceId, u16>,
}

impl BlockTextures {
    /// Decodes every texture of `content`. Textures that fail to decode are
    /// logged and replaced by the checkerboard, so content bugs show up in
    /// game instead of stopping it.
    pub fn load(content: &Content, max_layers: u32) -> Self {
        let mut base = missing_texture();
        let mut layer_of = HashMap::new();
        for (id, png) in content.textures() {
            if layer_of.len() + 1 >= max_layers as usize {
                warn!(
                    max_layers,
                    "too many block textures, the rest show as missing"
                );
                break;
            }
            match decode(png) {
                Ok(pixels) => {
                    layer_of.insert(id.clone(), layer_of.len() as u16 + 1);
                    base.extend_from_slice(&pixels);
                }
                Err(error) => warn!(%id, "broken texture: {error:#}"),
            }
        }
        // Some backends treat a single-layer array as a plain 2D texture.
        let mut layers = layer_of.len() as u32 + 1;
        if layers == 1 {
            base.extend_from_within(..);
            layers = 2;
        }
        Self {
            mips: mipmaps(base, layers),
            layers,
            layer_of,
        }
    }

    /// Layer of a texture; unknown textures get the checkerboard.
    pub fn layer(&self, id: &ResourceId) -> u16 {
        self.layer_of.get(id).copied().unwrap_or(0)
    }
}

fn decode(png: &[u8]) -> Result<Vec<u8>> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(png));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let size = reader.output_buffer_size().context("texture too large")?;
    let mut pixels = vec![0; size];
    let frame = reader.next_frame(&mut pixels)?;
    ensure!(
        (frame.width, frame.height) == (TEXTURE_SIZE, TEXTURE_SIZE),
        "expected {TEXTURE_SIZE}×{TEXTURE_SIZE} pixels, got {}×{}",
        frame.width,
        frame.height
    );
    pixels.truncate(frame.buffer_size());
    Ok(match frame.color_type {
        png::ColorType::Rgba => pixels,
        png::ColorType::Rgb => pixels
            .as_chunks::<3>()
            .0
            .iter()
            .flat_map(|&[r, g, b]| [r, g, b, 255])
            .collect(),
        png::ColorType::GrayscaleAlpha => pixels
            .as_chunks::<2>()
            .0
            .iter()
            .flat_map(|&[g, a]| [g, g, g, a])
            .collect(),
        png::ColorType::Grayscale => pixels.iter().flat_map(|&g| [g, g, g, 255]).collect(),
        png::ColorType::Indexed => unreachable!("EXPAND turns palettes into RGB(A)"),
    })
}

fn missing_texture() -> Vec<u8> {
    let half = TEXTURE_SIZE / 2;
    (0..TEXTURE_SIZE * TEXTURE_SIZE)
        .flat_map(|i| {
            let (x, y) = (i % TEXTURE_SIZE, i / TEXTURE_SIZE);
            if (x < half) == (y < half) {
                [255, 0, 255, 255]
            } else {
                [0, 0, 0, 255]
            }
        })
        .collect()
}

/// Halves every layer again and again by averaging 2×2 pixel blocks.
fn mipmaps(base: Vec<u8>, layers: u32) -> Vec<Vec<u8>> {
    let mut mips = vec![base];
    let mut size = TEXTURE_SIZE as usize;
    while size > 1 {
        let half = size / 2;
        let previous = mips.last().expect("at least the base level");
        let mut next = Vec::with_capacity(previous.len() / 4);
        for layer in previous.chunks_exact(size * size * 4).take(layers as usize) {
            for y in 0..half {
                for x in 0..half {
                    for channel in 0..4 {
                        let at = |dx: usize, dy: usize| {
                            u32::from(layer[((2 * y + dy) * size + 2 * x + dx) * 4 + channel])
                        };
                        next.push(((at(0, 0) + at(1, 0) + at(0, 1) + at(1, 1) + 2) / 4) as u8);
                    }
                }
            }
        }
        mips.push(next);
        size = half;
    }
    mips
}

#[cfg(test)]
mod tests {
    use ruda_core::ContentBuilder;

    use super::*;

    #[test]
    fn builds_every_mip_level_and_falls_back_to_the_checkerboard() {
        let mut content = ContentBuilder::new();
        content
            .add_texture("test:broken".parse().unwrap(), &b"not a png"[..])
            .unwrap();
        let textures = BlockTextures::load(&content.build(), 256);
        assert_eq!(textures.layers, 2);
        assert_eq!(textures.mips.len(), MIP_LEVELS as usize);
        for (level, mip) in textures.mips.iter().enumerate() {
            let size = (TEXTURE_SIZE >> level) as usize;
            assert_eq!(mip.len(), size * size * 4 * 2);
        }
        assert_eq!(textures.layer(&"test:broken".parse().unwrap()), 0);
    }
}
