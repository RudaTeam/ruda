//! The picture behind the main menu: a sunset in the manner of old 8-bit
//! games. It is drawn with big square pixels, in bands of a few colours, and
//! everything moves a step at a time rather than smoothly.

use std::time::Duration;

use egui::{Color32, Mesh, Painter, Rect, Shape, Ui, pos2, vec2};

use crate::widgets::Images;

/// The scene is this many pixels tall, whatever the window.
const ROWS: f32 = 190.0;
/// Everything moves in steps, this many a second.
const STEPS: f64 = 12.0;
/// A block of the ground is this many pixels a side, as a texture is.
const BLOCK: i64 = 16;
/// How many pixels a pixel of the sun's picture is.
const SUN_SCALE: f32 = 2.0;
/// Bands the sky is cut into.
const BANDS: i64 = 28;
/// Where the ground reaches up to, as a fraction of the height.
const HORIZON: f32 = 0.66;
/// Pixels a second the mountains and the ground slide by.
const MOUNTAIN_SPEED: f64 = 1.5;
const GROUND_SPEED: f64 = 4.0;

/// Colours of the sky from top to bottom, with where each is, from 0 to 1.
const SKY: [(f32, [u8; 3]); 5] = [
    (0.0, [29, 36, 86]),
    (0.36, [76, 63, 126]),
    (0.60, [176, 96, 122]),
    (0.82, [240, 163, 90]),
    (1.0, [247, 192, 122]),
];

/// A scrambled number for `n`, always the same for the same `n`.
fn hash(n: i64) -> u64 {
    let mut x = (n as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15);
    x ^= x >> 29;
    x = x.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    x ^ (x >> 32)
}

/// A number from 0 up to 1 for `n`, always the same for the same `n`.
fn unit(n: i64) -> f32 {
    (hash(n) >> 40) as f32 / (1u64 << 24) as f32
}

fn lerp(from: [u8; 3], to: [u8; 3], amount: f32) -> [u8; 3] {
    let mix = |a: u8, b: u8| (f32::from(a) + (f32::from(b) - f32::from(a)) * amount).round() as u8;
    [
        mix(from[0], to[0]),
        mix(from[1], to[1]),
        mix(from[2], to[2]),
    ]
}

/// The colour of the sky `fraction` of the way down.
fn sky(fraction: f32) -> [u8; 3] {
    let at = SKY
        .iter()
        .position(|&(stop, _)| stop >= fraction)
        .unwrap_or(SKY.len() - 1)
        .max(1);
    let ((from_stop, from), (to_stop, to)) = (SKY[at - 1], SKY[at]);
    lerp(
        from,
        to,
        ((fraction - from_stop) / (to_stop - from_stop)).clamp(0.0, 1.0),
    )
}

/// How tall the far mountain in column `column` is, in pixels.
fn mountain(column: i64) -> i64 {
    let n = column as f32 * 0.19;
    let height = 14.0 + 10.0 * (n.sin() + 0.6 * (n * 2.3 + 1.1).sin() + 0.3 * (n * 5.1).sin());
    // Even heights keep the steps blocky.
    (height.round() as i64).max(6) / 2 * 2
}

/// How many blocks tall the ground is at block column `column`.
fn ground(column: i64) -> i64 {
    let n = column as f32;
    let height = 3.5 + 1.6 * ((n * 0.45).sin() + 0.7 * (n * 1.3 + 2.0).sin());
    (height.round() as i64).clamp(2, 5)
}

/// Where scene pixels land on the screen.
struct Grid {
    origin: egui::Pos2,
    /// Points a pixel of the scene is wide: a whole number of the screen's
    /// own pixels, so the squares stay square.
    pixel: f32,
}

impl Grid {
    fn rect(&self, x: i64, y: i64, width: i64, height: i64) -> Rect {
        Rect::from_min_size(
            self.origin + vec2(x as f32, y as f32) * self.pixel,
            vec2(width as f32, height as f32) * self.pixel,
        )
    }
}

/// What the parts of the scene share while one frame of it is painted.
struct Scene<'a> {
    painter: Painter,
    images: &'a Images,
    grid: Grid,
    /// How many scene pixels wide and tall the screen is, a little over.
    columns: i64,
    rows: i64,
    /// Time in whole steps, and in seconds as of the step.
    step: i64,
    seconds: f64,
    /// How much of the scene shows, from 0 to 1.
    opacity: f32,
}

/// Paints the scene over `screen`, `opacity` there from 0 to 1.
pub(crate) fn paint(ui: &Ui, images: &Images, screen: Rect, opacity: f32) {
    let ctx = ui.ctx();
    let native = ctx.pixels_per_point();
    let pixel = ((screen.height() * native / ROWS).round().max(2.0)) / native;

    // Time goes in whole steps, and the next frame is due at the next one.
    let time = ui.input(|input| input.time);
    let step = (time * STEPS).floor() as i64;
    ctx.request_repaint_after(Duration::from_secs_f64(
        ((step + 1) as f64 / STEPS - time).max(0.001),
    ));

    let scene = Scene {
        painter: ui.painter().with_clip_rect(screen),
        images,
        grid: Grid {
            origin: screen.min,
            pixel,
        },
        columns: (screen.width() / pixel).ceil() as i64 + 1,
        rows: (screen.height() / pixel).ceil() as i64 + 1,
        step,
        seconds: step as f64 / STEPS,
        opacity,
    };
    scene.sky();
    scene.stars();
    scene.sun();
    scene.mountains();
    scene.clouds();
    scene.ground();
    scene.embers();
}

impl Scene<'_> {
    fn fade(&self, color: Color32) -> Color32 {
        color.gamma_multiply(self.opacity)
    }

    fn rgb(&self, color: [u8; 3]) -> Color32 {
        self.fade(Color32::from_rgb(color[0], color[1], color[2]))
    }

    /// The sky, in bands.
    fn sky(&self) {
        for band in 0..BANDS {
            let (top, bottom) = (self.rows * band / BANDS, self.rows * (band + 1) / BANDS);
            let color = self.rgb(sky((band as f32 + 0.5) / BANDS as f32));
            let rect = self.grid.rect(0, top, self.columns, bottom - top);
            self.painter.rect_filled(rect, 0.0, color);
        }
    }

    /// Stars in the upper sky, each twinkling at its own pace.
    fn stars(&self) {
        const BRIGHTNESS: [u8; 4] = [230, 130, 60, 130];
        for id in 0..36 {
            let x = (unit(id * 3) * self.columns as f32) as i64;
            let y = (unit(id * 3 + 1) * self.rows as f32 * 0.4) as i64;
            let phase = ((self.step / (5 + id % 5) + id) % 4) as usize;
            let color = Color32::from_rgba_unmultiplied(255, 245, 230, BRIGHTNESS[phase]);
            self.painter
                .rect_filled(self.grid.rect(x, y, 1, 1), 0.0, self.fade(color));
        }
    }

    /// The low sun, bobbing a pixel now and then and glowing in steps.
    fn sun(&self) {
        let Some(sun) = &self.images.sun else {
            return;
        };
        let size = (sun.size_vec2().x * SUN_SCALE) as i64;
        let bob = ((self.seconds * 0.35).sin() * 2.0).round() as i64;
        let x = (self.columns as f32 * 0.21) as i64 - size / 2;
        let y = (self.rows as f32 * 0.47) as i64 - size / 2 + bob;
        let glow = if (self.step / 6) % 2 == 0 { 1.0 } else { 0.88 };
        let uv = Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0));
        self.painter.image(
            sun.id(),
            self.grid.rect(x, y, size, size),
            uv,
            Color32::WHITE.gamma_multiply(self.opacity * glow),
        );
    }

    /// Far mountains in a bluish haze, as columns of a few pixels with steps
    /// between them, sliding slowly.
    fn mountains(&self) {
        const WIDTH: i64 = 6;
        let shift = (self.seconds * MOUNTAIN_SPEED).floor() as i64;
        let base = (self.rows as f32 * HORIZON) as i64;
        let (light, dark) = (self.rgb([96, 92, 150]), self.rgb([44, 47, 90]));
        let mut mesh = Mesh::default();
        for slot in -1..self.columns / WIDTH + 2 {
            let x = slot * WIDTH - shift.rem_euclid(WIDTH);
            let top = base - mountain((x + shift).div_euclid(WIDTH));
            let rect = self.grid.rect(x, top, WIDTH, self.rows - top);
            let first = mesh.vertices.len() as u32;
            mesh.colored_vertex(rect.left_top(), light);
            mesh.colored_vertex(rect.right_top(), light);
            mesh.colored_vertex(rect.right_bottom(), dark);
            mesh.colored_vertex(rect.left_bottom(), dark);
            mesh.add_triangle(first, first + 1, first + 2);
            mesh.add_triangle(first, first + 2, first + 3);
        }
        self.painter.add(Shape::mesh(mesh));
    }

    /// Blocky clouds, a few pieces each, drifting at different speeds.
    fn clouds(&self) {
        // Where each piece is in the cloud, and how big: x, y, width, height.
        const PIECES: [(i64, i64, i64, i64); 3] = [(0, 0, 44, 8), (8, -6, 24, 6), (14, 6, 30, 6)];
        let span = self.columns + 120;
        for id in 0..6 {
            let y = 10 + (unit(id * 5) * self.rows as f32 * 0.34) as i64;
            let speed = 1.0 + f64::from(unit(id * 5 + 1)) * 2.0;
            let start = (unit(id * 5 + 2) * span as f32) as i64;
            let x = (start + (self.seconds * speed).floor() as i64).rem_euclid(span) - 60;
            // Higher clouds are grey with violet, those near the horizon warm.
            let warmth = (y as f32 / (self.rows as f32 * HORIZON)).clamp(0.0, 1.0);
            let color = self.rgb(lerp([190, 172, 204], [238, 176, 150], warmth));
            for (dx, dy, width, height) in PIECES {
                let rect = self.grid.rect(x + dx, y + dy, width, height);
                self.painter
                    .rect_filled(rect, 0.0, color.gamma_multiply(0.55));
            }
        }
    }

    /// The ground in front: blocks of grass, dirt and stone, in the dusk's
    /// dim colours, sliding by as if the viewer walked.
    fn ground(&self) {
        let shift = (self.seconds * GROUND_SPEED).floor() as i64;
        let uv = Rect::from_min_max(egui::Pos2::ZERO, pos2(1.0, 1.0));
        for slot in -1..self.columns / BLOCK + 2 {
            let x = slot * BLOCK - shift.rem_euclid(BLOCK);
            let height = ground((x + shift).div_euclid(BLOCK));
            for cell in 0..height {
                let y = self.rows - (height - cell) * BLOCK;
                let (texture, tint) = match cell {
                    0 => (&self.images.grass, [176, 156, 170]),
                    1 | 2 => (&self.images.dirt, [124, 106, 134]),
                    _ => (&self.images.stone, [104, 98, 126]),
                };
                let tint = self.rgb(tint);
                let rect = self.grid.rect(x, y, BLOCK, BLOCK);
                if let Some(texture) = texture {
                    self.painter.image(texture.id(), rect, uv, tint);
                } else {
                    self.painter.rect_filled(rect, 0.0, tint);
                }
            }
        }
    }

    /// Sparks of ore rising from the ground, one pixel each, fading as they
    /// go.
    fn embers(&self) {
        const SWAY: [i64; 8] = [0, 0, 1, 1, 0, 0, -1, -1];
        let floor = (self.rows as f32 * 0.74) as i64;
        for id in 0..24 {
            let period = 60 + (unit(id * 7) * 60.0) as i64;
            let age = (self.step + (unit(id * 7 + 1) * period as f32) as i64) % period;
            let x =
                (unit(id * 7 + 2) * self.columns as f32) as i64 + SWAY[((age / 3) % 8) as usize];
            let alpha = (1.0 - age as f32 / period as f32) * 230.0;
            let color = Color32::from_rgba_unmultiplied(255, 150, 60, alpha as u8);
            let size = if id % 4 == 0 { 2 } else { 1 };
            let rect = self.grid.rect(x, floor - age / 2, size, size);
            self.painter.rect_filled(rect, 0.0, self.fade(color));
        }
    }
}

#[cfg(test)]
mod tests {
    use egui::{Context, RawInput};

    use super::*;

    /// What painting the scene at `time` asks for: its shapes, and how soon
    /// it wants to be painted again.
    fn frame(
        ctx: &Context,
        time: f64,
        screen: Rect,
    ) -> (Vec<egui::epaint::ClippedShape>, Duration) {
        let input = RawInput {
            screen_rect: Some(screen),
            time: Some(time),
            ..Default::default()
        };
        let mut output = ctx.run_ui(input, |ui| paint(ui, &Images::default(), screen, 1.0));
        output.textures_delta.clear();
        let delay = output
            .viewport_output
            .values()
            .map(|viewport| viewport.repaint_delay)
            .min()
            .unwrap_or(Duration::MAX);
        (output.shapes, delay)
    }

    fn screen() -> Rect {
        Rect::from_min_size(egui::Pos2::ZERO, vec2(1280.0, 720.0))
    }

    #[test]
    fn the_scene_moves_in_steps_and_asks_for_the_next_one() {
        let ctx = Context::default();
        let (first, delay) = frame(&ctx, 10.0, screen());
        assert!(!first.is_empty());
        // Within a step nothing moves, and the next is due before it ends.
        let (same, _) = frame(&ctx, 10.0 + 0.5 / STEPS, screen());
        assert_eq!(first, same);
        assert!(
            delay <= Duration::from_secs_f64(1.0 / STEPS + 0.005),
            "{delay:?}"
        );
        // A few seconds on, things have moved.
        let (later, _) = frame(&ctx, 14.0, screen());
        assert_ne!(first, later);
    }

    #[test]
    fn the_scene_fits_any_window() {
        for (width, height) in [
            (1280.0, 720.0),
            (640.0, 360.0),
            (3840.0, 2160.0),
            (300.0, 900.0),
        ] {
            let ctx = Context::default();
            let screen = Rect::from_min_size(egui::Pos2::ZERO, vec2(width, height));
            for time in [0.0, 1.0, 37.5] {
                assert!(!frame(&ctx, time, screen).0.is_empty());
            }
        }
    }

    #[test]
    fn the_sky_runs_from_deep_blue_to_warm_light() {
        let (top, bottom) = (sky(0.0), sky(1.0));
        assert_eq!(top, SKY[0].1);
        assert_eq!(bottom, SKY[SKY.len() - 1].1);
        assert!(top[2] > top[0], "dark blue at the top");
        assert!(bottom[0] > bottom[2], "warm at the horizon");
        // Between two stops it is between their colours.
        let middle = sky(0.18);
        assert!(middle[0] > top[0] && middle[0] < SKY[1].1[0]);
    }

    #[test]
    fn the_land_has_the_same_shape_every_time_and_stays_in_bounds() {
        for column in -500..500 {
            assert_eq!(mountain(column), mountain(column));
            assert!((6..=40).contains(&mountain(column)), "{}", mountain(column));
            assert_eq!(mountain(column) % 2, 0);
            assert!((2..=5).contains(&ground(column)));
        }
        // It is not flat.
        let heights: std::collections::HashSet<_> = (0..200).map(ground).collect();
        assert!(heights.len() >= 3);
    }
}
