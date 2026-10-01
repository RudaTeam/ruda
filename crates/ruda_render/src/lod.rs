//! Geometry for the far-away look of the world: the tops of cells, merged
//! where they match, and walls where the ground steps down.

use ruda_core::Face;
use ruda_world::lod::{LOD_TILE_CELLS, LodTile};

use crate::BlockFaces;

const CELLS: usize = LOD_TILE_CELLS;
/// How far walls on the edge of a tile reach down, hiding gaps to the
/// next tile, whose heights aren't known here.
const SKIRT: i32 = 8;

/// A box face of far-away terrain, packed into four words like a
/// [`crate::Quad`]:
///
/// - word 0: x and z of its first cell (6 bits each), its width along x and
///   depth along z in cells, minus one (6 bits each), face (3 bits);
/// - word 1: texture layer (10 bits), and 14 bits the renderer fills in at
///   bit 18;
/// - word 2: the bottom and top of the box in blocks, 16 bits each.
pub type LodQuad = [u32; 4];

/// The geometry of one tile.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct LodMesh {
    pub quads: Vec<LodQuad>,
    /// Lowest and highest point, for culling.
    pub min_y: i32,
    pub max_y: i32,
}

fn pack(
    x: usize,
    z: usize,
    width: usize,
    depth: usize,
    face: Face,
    layer: u16,
    (low, high): (i32, i32),
) -> LodQuad {
    [
        x as u32
            | (z as u32) << 6
            | (width as u32 - 1) << 12
            | (depth as u32 - 1) << 18
            | (face.index() as u32) << 24,
        u32::from(layer) & 0x3ff,
        (low as i16 as u16 as u32) | (high as i16 as u16 as u32) << 16,
        0,
    ]
}

pub fn mesh_lod(tile: &LodTile, faces: &BlockFaces) -> LodMesh {
    // The top surface of each cell: one above its top block.
    let top = |x: usize, z: usize| i32::from(tile.cell(x, z).0) + 1;
    let layer = |x: usize, z: usize, face: Face| {
        faces
            .layers(tile.cell(x, z).1)
            .map_or(0, |layers| layers[face.index()])
    };

    let mut quads = Vec::new();
    let (mut min_y, mut max_y) = (i32::MAX, i32::MIN);

    // Tops, merged into rectangles of equal height and texture.
    let mut cells = vec![[0u64; CELLS]; CELLS];
    for (z, row) in cells.iter_mut().enumerate() {
        for (x, cell) in row.iter_mut().enumerate() {
            let height = top(x, z);
            min_y = min_y.min(height - SKIRT);
            max_y = max_y.max(height);
            *cell = 1 << 63 | u64::from(height as u16) << 16 | u64::from(layer(x, z, Face::PosY));
        }
    }
    for z in 0..CELLS {
        let mut x = 0;
        while x < CELLS {
            let cell = cells[z][x];
            if cell == 0 {
                x += 1;
                continue;
            }
            let mut width = 1;
            while x + width < CELLS && cells[z][x + width] == cell {
                width += 1;
            }
            let mut depth = 1;
            while z + depth < CELLS && cells[z + depth][x..x + width].iter().all(|&c| c == cell) {
                depth += 1;
            }
            for row in &mut cells[z..z + depth] {
                row[x..x + width].fill(0);
            }
            let height = top(x, z);
            quads.push(pack(
                x,
                z,
                width,
                depth,
                Face::PosY,
                cell as u16,
                (height - 1, height),
            ));
            x += width;
        }
    }

    // Walls down to lower neighbours, merged along the edge they stand on.
    for face in [Face::PosX, Face::NegX, Face::PosZ, Face::NegZ] {
        let (dx, dz) = match face {
            Face::PosX => (1, 0),
            Face::NegX => (-1, 0),
            Face::PosZ => (0, 1),
            _ => (0, -1),
        };
        // `line` runs across the walls' direction, `step` along it.
        for line in 0..CELLS {
            let mut run: Option<(usize, (i32, i32), u16)> = None;
            for step in 0..=CELLS {
                let wall = (step < CELLS).then(|| {
                    let (x, z) = if dz == 0 { (line, step) } else { (step, line) };
                    let height = top(x, z);
                    let (nx, nz) = (x as i32 + dx, z as i32 + dz);
                    let inside = (0..CELLS as i32).contains(&nx) && (0..CELLS as i32).contains(&nz);
                    let low = if inside {
                        top(nx as usize, nz as usize)
                    } else {
                        height - SKIRT
                    };
                    (low < height).then(|| ((low, height), layer(x, z, face)))
                });
                let wall = wall.flatten();
                match (run, wall) {
                    (Some((_, span, texture)), Some((next_span, next_texture)))
                        if span == next_span && texture == next_texture => {}
                    _ => {
                        if let Some((start, span, texture)) = run.take() {
                            let length = step - start;
                            let (x, z, width, depth) = if dz == 0 {
                                (line, start, 1, length)
                            } else {
                                (start, line, length, 1)
                            };
                            quads.push(pack(x, z, width, depth, face, texture, span));
                        }
                        run = wall.map(|(span, texture)| (step, span, texture));
                    }
                }
            }
        }
    }
    LodMesh {
        quads,
        min_y,
        max_y,
    }
}

#[cfg(test)]
mod tests {
    use ruda_core::{Appearance, BlockDef, BlockId, ContentBuilder, CubeTextures, ResourceId};

    use super::*;

    fn faces() -> (BlockFaces, BlockId) {
        let mut content = ContentBuilder::new();
        let id: ResourceId = "test:grass".parse().unwrap();
        let grass = content
            .add_block(BlockDef::new(
                id.clone(),
                Appearance::Cube(CubeTextures::all(id)),
            ))
            .unwrap();
        let content = content.build();
        (BlockFaces::new(content.blocks(), |_| 5), grass)
    }

    #[test]
    fn a_flat_tile_is_one_top_and_skirts() {
        let (faces, grass) = faces();
        let tile = LodTile {
            heights: vec![10; CELLS * CELLS],
            blocks: vec![grass; CELLS * CELLS],
        };
        let mesh = mesh_lod(&tile, &faces);
        // One top, and one skirt along each edge.
        assert_eq!(mesh.quads.len(), 5);
        assert_eq!((mesh.min_y, mesh.max_y), (11 - SKIRT, 11));
        let top = mesh
            .quads
            .iter()
            .find(|q| (q[0] >> 24) & 7 == Face::PosY.index() as u32)
            .unwrap();
        assert_eq!(((top[0] >> 12) & 63, (top[0] >> 18) & 63), (63, 63));
        assert_eq!(top[2] >> 16, 11);
    }

    #[test]
    fn a_step_gets_a_wall() {
        let (faces, grass) = faces();
        let mut heights = vec![10; CELLS * CELLS];
        // The west half is 5 higher.
        for z in 0..CELLS {
            for x in 0..CELLS / 2 {
                heights[z * CELLS + x] = 15;
            }
        }
        let tile = LodTile {
            heights,
            blocks: vec![grass; CELLS * CELLS],
        };
        let mesh = mesh_lod(&tile, &faces);
        let wall = mesh
            .quads
            .iter()
            .find(|q| (q[0] >> 24) & 7 == Face::PosX.index() as u32 && q[0] & 63 == 31)
            .unwrap();
        // The whole edge in one wall, from the low ground to the high.
        assert_eq!((wall[0] >> 18) & 63, 63);
        assert_eq!((wall[2] & 0xffff, wall[2] >> 16), (11, 16));
    }
}
