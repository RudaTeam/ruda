//! Block textures: decoded from content PNGs into the layers of texture
//! arrays, with mipmaps for distant blocks. Each texture covers two blocks
//! by two, so a pattern that spans them doesn't repeat every block; smaller
//! textures are drawn once per block. Next to the colours, two more arrays
//! hold how each texture takes light, see [`ruda_core::TextureMaps`].

use std::collections::HashMap;

use anyhow::{Context as _, Result, ensure};
use ruda_core::{Content, ResourceId};
use tracing::warn;

/// Edge length of a texture array layer in pixels: two blocks.
pub const TEXTURE_SIZE: u32 = 64;
/// 64, 32, 16, 8, 4, 2 and 1 pixels.
pub const MIP_LEVELS: u32 = TEXTURE_SIZE.ilog2() + 1;
/// A flat surface, open to light all over.
const FLAT: [u8; 4] = [128, 128, 255, 0];
/// Matte, reflecting 4% straight back like most things, not glowing. The
/// alpha is how much it glows, 0 for not at all: unlike in labPBR, so that
/// mipmaps average it.
const MATTE: [u8; 4] = [0, 10, 0, 0];

/// Block textures as texture array layers. Layer 0 is a magenta checkerboard
/// that stands in for missing or broken textures.
#[derive(Clone, Debug)]
pub struct BlockTextures {
    /// RGBA pixels of every layer for each mip level, largest first.
    pub mips: Vec<Vec<u8>>,
    /// The same for normal maps: x and y of the surface's normal, ambient
    /// occlusion, and in alpha whether there is a map at all.
    pub normal_mips: Vec<Vec<u8>>,
    /// And for the rest: smoothness, reflectance, porosity and glow.
    pub specular_mips: Vec<Vec<u8>>,
    pub layers: u32,
    layer_of: HashMap<ResourceId, u16>,
}

impl BlockTextures {
    /// Decodes every texture of `content`. Textures that fail to decode are
    /// logged and replaced by the checkerboard, so content bugs show up in
    /// game instead of stopping it; broken maps are left out.
    pub fn load(content: &Content, max_layers: u32) -> Self {
        let pixels = (TEXTURE_SIZE * TEXTURE_SIZE) as usize;
        let mut colors = missing_texture();
        let mut normals = FLAT.repeat(pixels);
        let mut speculars = MATTE.repeat(pixels);
        let mut layer_of = HashMap::new();
        for (id, png) in content.textures() {
            if layer_of.len() + 1 >= max_layers as usize {
                warn!(
                    max_layers,
                    "too many block textures, the rest show as missing"
                );
                break;
            }
            let color = match decode(png) {
                Ok(color) => color,
                Err(error) => {
                    warn!(%id, "broken texture: {error:#}");
                    continue;
                }
            };
            let maps = content.texture_maps(id).and_then(|maps| {
                let decoded =
                    decode(&maps.normal).and_then(|normal| Ok((normal, decode(&maps.specular)?)));
                decoded
                    .inspect_err(|error| warn!(%id, "broken texture maps: {error:#}"))
                    .ok()
            });
            layer_of.insert(id.clone(), layer_of.len() as u16 + 1);
            colors.extend_from_slice(&color);
            match maps {
                Some((mut normal, mut specular)) => {
                    // Alpha says there is a map; labPBR keeps height there.
                    for pixel in normal.as_chunks_mut::<4>().0 {
                        pixel[3] = 255;
                    }
                    for pixel in specular.as_chunks_mut::<4>().0 {
                        pixel[3] = if pixel[3] == 255 { 0 } else { pixel[3] };
                    }
                    normals.extend_from_slice(&normal);
                    speculars.extend_from_slice(&specular);
                }
                None => {
                    normals.extend_from_slice(&FLAT.repeat(pixels));
                    speculars.extend_from_slice(&MATTE.repeat(pixels));
                }
            }
        }
        // Some backends treat a single-layer array as a plain 2D texture.
        let mut layers = layer_of.len() as u32 + 1;
        if layers == 1 {
            for array in [&mut colors, &mut normals, &mut speculars] {
                array.extend_from_within(..);
            }
            layers = 2;
        }
        Self {
            mips: mipmaps(colors, layers),
            normal_mips: mipmaps(normals, layers),
            specular_mips: mipmaps(speculars, layers),
            layers,
            layer_of,
        }
    }

    /// Layer of a texture; unknown textures get the checkerboard.
    pub fn layer(&self, id: &ResourceId) -> u16 {
        self.layer_of.get(id).copied().unwrap_or(0)
    }
}

/// A texture as a layer: as it is if it covers two blocks by two, or, if
/// it covers one, scaled to half a layer and repeated.
fn decode(png: &[u8]) -> Result<Vec<u8>> {
    let (width, height, pixels) = decode_image(png, TEXTURE_SIZE)?;
    ensure!(
        width == height && [16, 32, 64].contains(&width),
        "expected 16, 32 or 64 pixels square, got {width}×{height}"
    );
    if width == TEXTURE_SIZE {
        return Ok(pixels);
    }
    let half = TEXTURE_SIZE / 2;
    let scale = half / width;
    let mut layer = Vec::with_capacity((TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize);
    for y in 0..TEXTURE_SIZE {
        for x in 0..TEXTURE_SIZE {
            let (sx, sy) = ((x % half) / scale, (y % half) / scale);
            let at = ((sy * width + sx) * 4) as usize;
            layer.extend_from_slice(&pixels[at..at + 4]);
        }
    }
    Ok(layer)
}

/// Width, height and RGBA pixels of a PNG image at most `max_size` pixels
/// on a side.
pub(crate) fn decode_image(png: &[u8], max_size: u32) -> Result<(u32, u32, Vec<u8>)> {
    let mut decoder = png::Decoder::new(std::io::Cursor::new(png));
    decoder.set_transformations(png::Transformations::EXPAND | png::Transformations::STRIP_16);
    let mut reader = decoder.read_info()?;
    let size = reader.output_buffer_size().context("texture too large")?;
    let mut pixels = vec![0; size];
    let frame = reader.next_frame(&mut pixels)?;
    ensure!(
        frame.width <= max_size && frame.height <= max_size,
        "images can be at most {max_size}×{max_size} pixels, this one is {}×{}",
        frame.width,
        frame.height
    );
    pixels.truncate(frame.buffer_size());
    let rgba = match frame.color_type {
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
    };
    Ok((frame.width, frame.height, rgba))
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
        for mips in [
            &textures.mips,
            &textures.normal_mips,
            &textures.specular_mips,
        ] {
            assert_eq!(mips.len(), MIP_LEVELS as usize);
            for (level, mip) in mips.iter().enumerate() {
                let size = (TEXTURE_SIZE >> level) as usize;
                assert_eq!(mip.len(), size * size * 4 * 2);
            }
        }
        assert_eq!(textures.layer(&"test:broken".parse().unwrap()), 0);
    }

    #[test]
    fn a_texture_for_one_block_repeats_two_by_two() {
        // 16 pixels, red in the corner and blue elsewhere.
        let mut pixels = [0, 0, 255, 255].repeat(16 * 16);
        pixels[..4].copy_from_slice(&[255, 0, 0, 255]);
        let mut png = Vec::new();
        let mut encoder = png::Encoder::new(&mut png, 16, 16);
        encoder.set_color(png::ColorType::Rgba);
        encoder.set_depth(png::BitDepth::Eight);
        encoder
            .write_header()
            .unwrap()
            .write_image_data(&pixels)
            .unwrap();
        let mut content = ContentBuilder::new();
        let id: ResourceId = "test:tile".parse().unwrap();
        content.add_texture(id.clone(), png).unwrap();
        let textures = BlockTextures::load(&content.build(), 256);
        let layer = &textures.mips[0][(TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize..];
        let red = |x: u32, y: u32| layer[((y * TEXTURE_SIZE + x) * 4) as usize] == 255;
        // Scaled to 32 pixels, so the corner pixel is two by two, and
        // repeated: the next block starts at 32.
        assert!(red(0, 0) && red(1, 1) && !red(2, 0));
        assert!(red(32, 0) && red(0, 32) && red(33, 33));
        // Without maps, a texture is flat and matte.
        let normal = &textures.normal_mips[0][(TEXTURE_SIZE * TEXTURE_SIZE * 4) as usize..];
        assert_eq!(normal[..4], FLAT);
    }
}
