//! Pieces drawn by hand in the game's own look: stone with a glow of ore.

use egui::epaint::Shadow;
use egui::{
    Color32, Mesh, Painter, Pos2, Rect, Response, Sense, Shape, Stroke, StrokeKind, TextureFilter,
    TextureHandle, TextureOptions, TextureWrapMode, Ui, Vec2, WidgetInfo, WidgetType,
    emath::GuiRounding, pos2, vec2,
};

use crate::style::{ACCENT, ACCENT_LIGHT, OUTLINE, TEXT, pixel};

/// The texture pieces are tiled from have 16 texels a side; each is drawn
/// this many points wide.
pub(crate) const TEXEL: f32 = 2.0;
const TILE: f32 = 16.0 * TEXEL;
/// Thickness of the outline and the bevel of stone pieces.
pub(crate) const EDGE: f32 = 2.0 * TEXEL;

pub const BUTTON_SIZE: Vec2 = Vec2::new(360.0, 56.0);
/// Between buttons that are laid out together.
pub(crate) const BUTTON_GAP: f32 = 16.0;
/// Between the edge of a panel and what is on it.
pub(crate) const PANEL_MARGIN: f32 = 24.0;

/// For pictures drawn once at any size.
pub const PICTURE_TEXTURE: TextureOptions = TextureOptions::LINEAR;
/// For the small textures that stone is tiled from: crisp, repeating.
pub const TILING_TEXTURE: TextureOptions = TextureOptions {
    magnification: TextureFilter::Nearest,
    minification: TextureFilter::Nearest,
    wrap_mode: TextureWrapMode::Repeat,
    mipmap_mode: None,
};

/// The images the menus are drawn with. Whatever is missing is replaced by
/// flat colour.
#[derive(Clone, Default)]
pub struct Images {
    pub logo: Option<TextureHandle>,
    pub cobblestone: Option<TextureHandle>,
    /// What the sunset behind the main menu is made of: the sun, and the
    /// blocks of the ground.
    pub sun: Option<TextureHandle>,
    pub grass: Option<TextureHandle>,
    pub dirt: Option<TextureHandle>,
    pub stone: Option<TextureHandle>,
}

impl std::fmt::Debug for Images {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Images")
            .field("logo", &self.logo.is_some())
            .field("cobblestone", &self.cobblestone.is_some())
            .field("sun", &self.sun.is_some())
            .field("grass", &self.grass.is_some())
            .field("dirt", &self.dirt.is_some())
            .field("stone", &self.stone.is_some())
            .finish()
    }
}

/// A piece of stone: outlined, tiled from cobblestone and bevelled, lit up
/// orange as `glow` goes from 0 to 1.
pub(crate) fn stone_piece(
    painter: &Painter,
    images: &Images,
    rect: Rect,
    glow: f32,
    pressed: bool,
    enabled: bool,
) {
    if glow > 0.0 {
        painter.add(
            Shadow {
                offset: [0, 0],
                blur: 28,
                spread: 4,
                color: ACCENT.gamma_multiply(0.5 * glow),
            }
            .as_shape(rect, 0.0),
        );
    }
    painter.rect_stroke(
        rect,
        0.0,
        Stroke::new(EDGE, OUTLINE.lerp_to_gamma(ACCENT, glow)),
        StrokeKind::Outside,
    );

    let tint = if enabled {
        let normal = Color32::from_rgb(150, 154, 164);
        let warm = Color32::from_rgb(198, 162, 136);
        let tint = normal.lerp_to_gamma(warm, glow);
        if pressed {
            tint.gamma_multiply(0.8)
        } else {
            tint
        }
    } else {
        Color32::from_rgb(88, 90, 98)
    };
    tiled(painter, images.cobblestone.as_ref(), rect, tint);

    let (light, dark) = if pressed {
        (
            Color32::from_black_alpha(130),
            Color32::from_white_alpha(50),
        )
    } else {
        (
            Color32::from_white_alpha(56),
            Color32::from_black_alpha(130),
        )
    };
    let light = light.lerp_to_gamma(ACCENT_LIGHT.gamma_multiply(0.45), glow * 0.6);
    bevel(painter, rect, light, dark);
}

/// Strips along the edges of `rect`, `light` on the top and left and `dark`
/// on the bottom and right.
pub(crate) fn bevel(painter: &Painter, rect: Rect, light: Color32, dark: Color32) {
    let inner = Rect::from_min_max(rect.min + Vec2::splat(EDGE), rect.max - Vec2::splat(EDGE));
    for (strip, color) in [
        (
            Rect::from_min_max(rect.min, pos2(rect.max.x, inner.min.y)),
            light,
        ),
        (
            Rect::from_min_max(
                pos2(rect.min.x, inner.min.y),
                pos2(inner.min.x, inner.max.y),
            ),
            light,
        ),
        (
            Rect::from_min_max(pos2(rect.min.x, inner.max.y), rect.max),
            dark,
        ),
        (
            Rect::from_min_max(
                pos2(inner.max.x, inner.min.y),
                pos2(rect.max.x, inner.max.y),
            ),
            dark,
        ),
    ] {
        painter.rect_filled(strip, 0.0, color);
    }
}

/// A large panel of dark stone for menus to sit on.
pub(crate) fn panel(painter: &Painter, images: &Images, rect: Rect) {
    painter.rect_stroke(rect, 0.0, Stroke::new(EDGE, OUTLINE), StrokeKind::Outside);
    tiled(
        painter,
        images.cobblestone.as_ref(),
        rect,
        Color32::from_rgb(66, 68, 78),
    );
    bevel(
        painter,
        rect,
        Color32::from_white_alpha(40),
        Color32::from_black_alpha(150),
    );
}

/// A dark hollow pressed into a panel, for what is read or set there.
pub(crate) fn sunken(painter: &Painter, rect: Rect) {
    painter.rect_filled(rect, 0.0, Color32::from_rgb(14, 15, 19));
    bevel(
        painter,
        rect,
        Color32::from_black_alpha(170),
        Color32::from_white_alpha(26),
    );
}

/// Text in the pixel font centred on `center`, with the dark shadow the
/// pieces' labels have.
pub(crate) fn pixel_label(
    painter: &Painter,
    center: Pos2,
    text: &str,
    color: Color32,
    shadow: bool,
) {
    pixel_text(painter, center, text, 24.0, color, shadow, 1.0);
}

/// Pixel font text of `size` points that is `opacity` there, from 0 to 1.
pub(crate) fn pixel_text(
    painter: &Painter,
    center: Pos2,
    text: &str,
    size: f32,
    color: Color32,
    shadow: bool,
    opacity: f32,
) {
    let label = painter.layout_no_wrap(text.to_owned(), pixel(size), color.gamma_multiply(opacity));
    let at = (center - label.size() / 2.0).round_to_pixels(painter.pixels_per_point());
    if shadow {
        let dark = Color32::from_black_alpha(166).gamma_multiply(opacity);
        let dark = painter.layout_no_wrap(text.to_owned(), pixel(size), dark);
        painter.galley(at + Vec2::splat(TEXEL), dark, Color32::PLACEHOLDER);
    }
    painter.galley(at, label, Color32::PLACEHOLDER);
}

/// The title of a screen: large pixel text.
pub(crate) fn heading(painter: &Painter, center: Pos2, text: &str) {
    pixel_text(painter, center, text, 40.0, TEXT, true, 1.0);
}

/// How lit up a widget is: 0 to 1, eased over a moment, for the pointer or
/// the keyboard being on it.
pub(crate) fn glow(ui: &Ui, response: &Response, enabled: bool) -> f32 {
    let lit = enabled && (response.hovered() || response.has_focus());
    ui.ctx()
        .animate_bool_with_time(response.id.with("glow"), lit, 0.12)
}

/// A button that looks like a block of stone and glows when it is the one
/// the pointer or the keyboard is on.
pub fn stone_button(ui: &mut Ui, images: &Images, text: &str, enabled: bool) -> Response {
    stone_button_sized(ui, images, text, BUTTON_SIZE, enabled)
}

pub fn stone_button_sized(
    ui: &mut Ui,
    images: &Images,
    text: &str,
    size: Vec2,
    enabled: bool,
) -> Response {
    stone_button_lit(ui, images, text, size, enabled, false)
}

/// A stone button that stays lit up, to show it is the one chosen among a
/// few, as well as when the pointer or the keyboard is on it.
pub fn stone_button_lit(
    ui: &mut Ui,
    images: &Images,
    text: &str,
    size: Vec2,
    enabled: bool,
    lit: bool,
) -> Response {
    let sense = if enabled {
        Sense::click()
    } else {
        Sense::hover()
    };
    let (rect, response) = ui.allocate_exact_size(size, sense);
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, enabled, text));
    if !ui.is_rect_visible(rect) {
        return response;
    }
    let rect = rect.round_to_pixels(ui.pixels_per_point());
    let glow = glow(ui, &response, enabled).max(if lit { 1.0 } else { 0.0 });
    let pressed = enabled && response.is_pointer_button_down_on();
    stone_piece(ui.painter(), images, rect, glow, pressed, enabled);

    let color = if !enabled {
        TEXT.gamma_multiply(0.45)
    } else {
        TEXT.lerp_to_gamma(ACCENT_LIGHT, glow)
    };
    let press = if pressed {
        vec2(0.0, TEXEL)
    } else {
        Vec2::ZERO
    };
    pixel_label(ui.painter(), rect.center() + press, text, color, enabled);
    response
}

/// A square stone button with a picture on it, drawn by `icon` around the
/// middle of the button, given how lit the button is.
pub fn stone_icon_button(
    ui: &mut Ui,
    images: &Images,
    size: Vec2,
    label: &str,
    icon: impl FnOnce(&Painter, Pos2, f32),
) -> Response {
    let (rect, response) = ui.allocate_exact_size(size, Sense::click());
    response.widget_info(|| WidgetInfo::labeled(WidgetType::Button, true, label));
    if ui.is_rect_visible(rect) {
        let rect = rect.round_to_pixels(ui.pixels_per_point());
        let glow = glow(ui, &response, true);
        let pressed = response.is_pointer_button_down_on();
        stone_piece(ui.painter(), images, rect, glow, pressed, true);
        let press = if pressed {
            vec2(0.0, TEXEL)
        } else {
            Vec2::ZERO
        };
        icon(ui.painter(), rect.center() + press, glow);
    }
    response
}

/// The globe on the language button, row by row: `o` the outline, `b` and
/// `B` the sea in light and in shade, `l` a glint, `g` and `G` land.
const GLOBE: [&str; 14] = [
    "....oooooo....",
    "..oolhbbbboo..",
    ".olhhbbggbbbo.",
    ".olhbbgggbbbo.",
    "olbbbggggbbggo",
    "obbbbgggbbgggo",
    "obbbbbggbbgGgo",
    "obbbbbbbbbbGBo",
    "obbggbbbbbbBBo",
    "obgggbbbbggBBo",
    ".obggbbbbggBo.",
    ".obbbbbbbbBBo.",
    "..ooBBBBBBoo..",
    "....oooooo....",
];

/// Draws the globe, each of its pixels this many points a side.
pub(crate) fn paint_globe(painter: &Painter, center: Pos2, pixel: f32) {
    let colour = |c: char| match c {
        'o' => Color32::from_rgb(16, 22, 48),
        'b' => Color32::from_rgb(52, 112, 200),
        'B' => Color32::from_rgb(34, 78, 152),
        'l' => Color32::from_rgb(150, 196, 244),
        'h' => Color32::from_rgb(200, 226, 255),
        'g' => Color32::from_rgb(92, 170, 78),
        _ => Color32::from_rgb(54, 122, 58),
    };
    let size = GLOBE.len() as f32 * pixel;
    let origin = center - Vec2::splat(size / 2.0);
    for (row, line) in GLOBE.iter().enumerate() {
        // Runs of one colour are drawn as one piece.
        let cells: Vec<char> = line.chars().collect();
        let mut start = 0;
        while start < cells.len() {
            let c = cells[start];
            let mut end = start;
            while end < cells.len() && cells[end] == c {
                end += 1;
            }
            if c != '.' {
                let at = origin + vec2(start as f32, row as f32) * pixel;
                let rect = Rect::from_min_size(at, vec2((end - start) as f32 * pixel, pixel));
                painter.rect_filled(
                    rect.round_to_pixels(painter.pixels_per_point()),
                    0.0,
                    colour(c),
                );
            }
            start = end;
        }
    }
}

/// A bar with a lit piece sliding along it, for waiting without knowing how
/// long. `time` is in seconds.
pub fn loading_bar(painter: &Painter, center: Pos2, time: f64) {
    let size = vec2(BUTTON_SIZE.x, 8.0 * TEXEL);
    let track = Rect::from_center_size(center, size).round_to_pixels(painter.pixels_per_point());
    let rim = Stroke::new(EDGE, Color32::from_rgb(52, 56, 66));
    painter.rect_stroke(track, 0.0, rim, StrokeKind::Outside);
    painter.rect_filled(track, 0.0, Color32::from_rgb(11, 12, 15));

    // Slides in from one side and out of the other, a texel at a time.
    let piece = track.width() * 0.3;
    let phase = (time / 1.6).fract() as f32;
    let left = track.min.x - piece + phase * (track.width() + piece);
    let left = (left / TEXEL).round() * TEXEL;
    let lit = Rect::from_min_size(pos2(left, track.min.y), vec2(piece, track.height()));
    let painter = painter.with_clip_rect(track);
    painter.rect_filled(lit, 0.0, ACCENT);
    let top = Rect::from_min_size(lit.min, vec2(piece, EDGE / 2.0));
    painter.rect_filled(top, 0.0, ACCENT_LIGHT.gamma_multiply(0.6));
}

/// Fills `rect` with `texture` repeated at the size of the game's pixels, or
/// with plain `tint` if there is none.
pub fn tiled(painter: &egui::Painter, texture: Option<&TextureHandle>, rect: Rect, tint: Color32) {
    match texture {
        Some(texture) => {
            let uv = Rect::from_min_size(Pos2::ZERO, rect.size() / TILE);
            painter.image(texture.id(), rect, uv, tint);
        }
        None => {
            painter.rect_filled(rect, 0.0, tint);
        }
    }
}

/// Paints the sunset scene over `screen`, `scene` showing from 0 to 1, then
/// darkens `screen` towards the edges so menus on top stay readable, over
/// the scene or over whatever else is drawn behind.
pub fn backdrop(ui: &Ui, images: &Images, screen: Rect, scene: f32) {
    if scene > 0.0 {
        crate::scene::paint(ui, images, screen, scene);
    }
    let painter = ui.painter();
    // The scene is brighter than a world at dusk, so it needs less.
    let (centre, edge) = (0.3 - 0.18 * scene, 0.85 - 0.30 * scene);

    // A grid of vertices whose darkness grows with the distance from the
    // middle, smoothed out by the interpolation between them.
    const COLUMNS: usize = 24;
    const ROWS: usize = 14;
    let mut mesh = Mesh::default();
    for row in 0..=ROWS {
        for column in 0..=COLUMNS {
            let (x, y) = (column as f32 / COLUMNS as f32, row as f32 / ROWS as f32);
            let from_middle = vec2(x - 0.5, y - 0.5) * 2.0;
            let distance = (from_middle.length() / std::f32::consts::SQRT_2).clamp(0.0, 1.0);
            let alpha = centre + (edge - centre) * distance * distance;
            let at = pos2(
                screen.min.x + x * screen.width(),
                screen.min.y + y * screen.height(),
            );
            mesh.colored_vertex(at, Color32::from_black_alpha((alpha * 255.0) as u8));
        }
    }
    let at = |row: usize, column: usize| (row * (COLUMNS + 1) + column) as u32;
    for row in 0..ROWS {
        for column in 0..COLUMNS {
            let (a, b) = (at(row, column), at(row, column + 1));
            let (c, d) = (at(row + 1, column), at(row + 1, column + 1));
            mesh.add_triangle(a, b, c);
            mesh.add_triangle(b, d, c);
        }
    }
    painter.add(Shape::mesh(mesh));
}
