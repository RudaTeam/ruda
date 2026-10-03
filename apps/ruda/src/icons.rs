//! Pictures of blocks for the interface: a cube seen from a corner, built
//! from the textures of its faces, or a torch's own picture.

use egui::ColorImage;
use ruda_core::{Appearance, BlockId, Content, ResourceId};

use crate::decode_png;

/// Icons are this many pixels square.
const SIZE: usize = 64;
/// Samples taken along each side of a pixel, to smooth the cube's outline.
const SAMPLES: usize = 2;
/// How much of its light each face of a cube gets: the top is lit, the left
/// a little less, the right the least.
const SHADES: [f32; 3] = [1.0, 0.8, 0.6];

/// A decoded texture.
struct Texture {
    width: usize,
    height: usize,
    rgba: Vec<u8>,
}

impl Texture {
    fn load(content: &Content, id: &ResourceId) -> Option<Self> {
        let (width, height, rgba) = decode_png(content.texture(id)?).ok()?;
        Some(Self {
            width: width as usize,
            height: height as usize,
            rgba,
        })
    }

    /// The pixel at `(u, v)`, each from 0 to 1.
    fn sample(&self, u: f32, v: f32) -> [u8; 4] {
        let x = ((u * self.width as f32) as usize).min(self.width - 1);
        let y = ((v * self.height as f32) as usize).min(self.height - 1);
        let at = (y * self.width + x) * 4;
        [
            self.rgba[at],
            self.rgba[at + 1],
            self.rgba[at + 2],
            self.rgba[at + 3],
        ]
    }
}

/// The picture of `block`, if it has a look to show.
pub fn block_icon(content: &Content, block: BlockId) -> Option<ColorImage> {
    match &content.blocks().get(block).appearance {
        Appearance::Invisible => None,
        Appearance::Cube(textures) => {
            let top = Texture::load(content, &textures.top)?;
            let side = Texture::load(content, &textures.side)?;
            Some(cube(&top, &side))
        }
        Appearance::Torch { texture } => Some(sprite(&Texture::load(content, texture)?)),
    }
}

/// A corner of the cube, in the picture's units: from -1 to 1 either way.
type Point = (f32, f32);

/// A face of the cube as a parallelogram: where the texture's corner is, and
/// where its sides go.
struct Face {
    origin: Point,
    across: Point,
    down: Point,
}

impl Face {
    /// Where `point` is on the face, as `(u, v)` from 0 to 1 if it is on it.
    fn at(&self, point: Point) -> Option<(f32, f32)> {
        let (px, py) = (point.0 - self.origin.0, point.1 - self.origin.1);
        let (ax, ay) = self.across;
        let (dx, dy) = self.down;
        let det = ax * dy - ay * dx;
        let u = (px * dy - py * dx) / det;
        let v = (ax * py - ay * px) / det;
        ((0.0..1.0).contains(&u) && (0.0..1.0).contains(&v)).then_some((u, v))
    }
}

/// A cube with its top and two sides in view.
fn cube(top: &Texture, side: &Texture) -> ColorImage {
    const W: f32 = 0.866_025_4;
    let faces = [
        (
            top,
            Face {
                origin: (-W, -0.5),
                across: (W, -0.5),
                down: (W, 0.5),
            },
            SHADES[0],
        ),
        (
            side,
            Face {
                origin: (-W, -0.5),
                across: (W, 0.5),
                down: (0.0, 1.0),
            },
            SHADES[1],
        ),
        (
            side,
            Face {
                origin: (0.0, 0.0),
                across: (W, -0.5),
                down: (0.0, 1.0),
            },
            SHADES[2],
        ),
    ];
    let mut rgba = vec![0u8; SIZE * SIZE * 4];
    for y in 0..SIZE {
        for x in 0..SIZE {
            let (mut sum, mut alpha) = ([0.0f32; 3], 0.0f32);
            for sy in 0..SAMPLES {
                for sx in 0..SAMPLES {
                    let fx = (x as f32 + (sx as f32 + 0.5) / SAMPLES as f32) / SIZE as f32;
                    let fy = (y as f32 + (sy as f32 + 0.5) / SAMPLES as f32) / SIZE as f32;
                    let point = (fx * 2.0 - 1.0, fy * 2.0 - 1.0);
                    let hit = faces
                        .iter()
                        .find_map(|(texture, face, shade)| Some((texture, face.at(point)?, shade)));
                    if let Some((texture, (u, v), shade)) = hit {
                        let [r, g, b, a] = texture.sample(u, v);
                        let a = f32::from(a) / 255.0;
                        for (sum, channel) in sum.iter_mut().zip([r, g, b]) {
                            *sum += f32::from(channel) * shade * a;
                        }
                        alpha += a;
                    }
                }
            }
            if alpha > 0.0 {
                let at = (y * SIZE + x) * 4;
                for (out, sum) in rgba[at..at + 3].iter_mut().zip(sum) {
                    *out = (sum / alpha).round().min(255.0) as u8;
                }
                rgba[at + 3] = (alpha / (SAMPLES * SAMPLES) as f32 * 255.0).round() as u8;
            }
        }
    }
    ColorImage::from_rgba_unmultiplied([SIZE, SIZE], &rgba)
}

/// A flat picture, such as a torch's, scaled to fill the icon: magnified in
/// whole pixels when it is small, as the pixels of a 16 pixel texture should
/// stay square, and shrunk when it is larger than the icon.
fn sprite(texture: &Texture) -> ColorImage {
    let scale = SIZE as f32 / texture.width.max(texture.height) as f32;
    let scale = if scale >= 1.0 { scale.floor() } else { scale };
    let width = ((texture.width as f32 * scale).round() as usize).clamp(1, SIZE);
    let height = ((texture.height as f32 * scale).round() as usize).clamp(1, SIZE);
    let (left, top) = ((SIZE - width) / 2, (SIZE - height) / 2);
    let mut rgba = vec![0u8; SIZE * SIZE * 4];
    for y in 0..height {
        for x in 0..width {
            let u = (x as f32 + 0.5) / width as f32;
            let v = (y as f32 + 0.5) / height as f32;
            let to = ((top + y) * SIZE + left + x) * 4;
            rgba[to..to + 4].copy_from_slice(&texture.sample(u, v));
        }
    }
    ColorImage::from_rgba_unmultiplied([SIZE, SIZE], &rgba)
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use ruda_core::ContentBuilder;

    use super::*;

    fn content() -> Arc<Content> {
        let mut content = ContentBuilder::new();
        ruda_base::register(&mut content).unwrap();
        Arc::new(content.build())
    }

    fn alpha(image: &ColorImage, x: usize, y: usize) -> u8 {
        image.pixels[y * SIZE + x].a()
    }

    #[test]
    fn every_block_a_player_can_place_has_an_icon() {
        let content = content();
        let blocks = crate::game::placeable_blocks(&content);
        assert!(blocks.len() >= 10);
        for block in blocks {
            let name = content.blocks().get(block).id.as_str();
            assert!(block_icon(&content, block).is_some(), "{name}");
        }
    }

    #[test]
    fn a_cube_fills_the_middle_and_leaves_the_corners() {
        let content = content();
        let stone = content
            .blocks()
            .id(&ruda_base::id("stone").unwrap())
            .unwrap();
        let icon = block_icon(&content, stone).unwrap();
        assert_eq!(alpha(&icon, SIZE / 2, SIZE / 2), 255);
        assert_eq!(alpha(&icon, 1, 1), 0);
        assert_eq!(alpha(&icon, SIZE - 2, SIZE - 2), 0);
        // The top is lighter than the sides, the left lighter than the right.
        let brightness = |x: usize, y: usize| {
            let pixel = icon.pixels[y * SIZE + x];
            u32::from(pixel.r()) + u32::from(pixel.g()) + u32::from(pixel.b())
        };
        let (top, left, right) = (
            brightness(SIZE / 2, SIZE / 4),
            brightness(SIZE / 4, SIZE * 5 / 8),
            brightness(SIZE * 3 / 4, SIZE * 5 / 8),
        );
        assert!(top > left && left > right, "{top} {left} {right}");
    }

    #[test]
    fn sprites_of_any_size_fill_the_icon_without_leaving_it() {
        let texture = |size: usize| Texture {
            width: size,
            height: size,
            rgba: [200, 100, 50, 255].repeat(size * size),
        };
        // 16 pixels are magnified four times over the whole icon.
        let small = sprite(&texture(16));
        assert!(small.pixels.iter().all(|pixel| pixel.a() == 255));
        // 128 pixels are shrunk, not cropped to a corner.
        let large = sprite(&texture(128));
        assert!(large.pixels.iter().all(|pixel| pixel.a() == 255));
        // A sprite that is not square stays centred.
        let wide = Texture {
            width: 32,
            height: 16,
            rgba: [10, 20, 30, 255].repeat(32 * 16),
        };
        let wide = sprite(&wide);
        assert_eq!(alpha(&wide, SIZE / 2, SIZE / 2), 255);
        assert_eq!(alpha(&wide, SIZE / 2, 1), 0);
    }

    #[test]
    fn a_torch_is_its_own_picture() {
        let content = content();
        let torch = content
            .blocks()
            .id(&ruda_base::id("torch").unwrap())
            .unwrap();
        let icon = block_icon(&content, torch).unwrap();
        // The stick is in the middle, and there is nothing at the sides.
        assert!(alpha(&icon, SIZE / 2, SIZE / 2) > 0);
        assert_eq!(alpha(&icon, 2, SIZE / 2), 0);
    }
}
