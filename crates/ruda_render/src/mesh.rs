//! Turning chunks into geometry: one packed quad per visible run of faces,
//! with the light and ambient occlusion at each of its corners.

use ruda_core::{
    Appearance, BlockId, BlockRegistry, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, Face, Light, LocalPos,
    ResourceId,
};
use ruda_world::World;

use crate::Visibility;

const SIZE: usize = CHUNK_SIZE as usize;
/// A chunk plus a one-block border on every side.
const PADDED: usize = SIZE + 2;

/// For every block, the texture layer of each face, or `None` for blocks
/// that are not drawn, and whether it glows.
#[derive(Clone, Debug, Default)]
pub struct BlockFaces {
    faces: Vec<Option<[u16; 6]>>,
    glowing: Vec<bool>,
}

impl BlockFaces {
    /// `layer` maps a texture to its layer in the block texture array.
    pub fn new(blocks: &BlockRegistry, layer: impl Fn(&ResourceId) -> u16) -> Self {
        let faces = blocks
            .iter()
            .map(|(_, def)| match &def.appearance {
                Appearance::Invisible => None,
                Appearance::Cube(textures) => {
                    Some(Face::ALL.map(|face| layer(textures.for_face(face))))
                }
            })
            .collect();
        let glowing = blocks.iter().map(|(_, def)| !def.light.is_dark()).collect();
        Self { faces, glowing }
    }

    fn get(&self, id: BlockId) -> Option<[u16; 6]> {
        match self.faces.get(id.index()) {
            Some(faces) => *faces,
            None => self.faces.get(BlockId::UNKNOWN.index()).copied().flatten(),
        }
    }

    fn is_solid(&self, id: BlockId) -> bool {
        self.get(id).is_some()
    }

    fn glows(&self, id: BlockId) -> bool {
        self.glowing.get(id.index()).copied().unwrap_or(false)
    }
}

/// A chunk's blocks and light with a one-block border copied from the 26
/// chunks around it: enough to tell which faces touch air and how light
/// and shadow fall on them.
#[derive(Clone, Debug)]
pub struct PaddedChunk {
    blocks: Box<[BlockId]>,
    light: Box<[Light]>,
}

impl PaddedChunk {
    /// `None` if the chunk itself is not loaded. Neighbours that are not
    /// loaded count as solid and dark, so faces towards them stay hidden
    /// until they arrive and the chunk is meshed again.
    pub fn gather(world: &World, pos: ChunkPos) -> Option<Self> {
        let chunk = world.chunk(pos)?;
        let mut blocks = vec![BlockId::UNKNOWN; PADDED * PADDED * PADDED].into_boxed_slice();
        let mut light = vec![Light::DARK; PADDED * PADDED * PADDED].into_boxed_slice();

        let mut inner_blocks = vec![BlockId::AIR; CHUNK_VOLUME];
        chunk.copy_to(&mut inner_blocks);
        let mut inner_light = vec![Light::DARK; CHUNK_VOLUME];
        chunk.light().copy_to(&mut inner_light);
        for index in 0..CHUNK_VOLUME {
            let local = LocalPos::from_index(index);
            let padded = padded_index([local.x(), local.y(), local.z()].map(|c| c as usize + 1));
            blocks[padded] = inner_blocks[index];
            light[padded] = inner_light[index];
        }

        // The layers, edges and corners of the 26 neighbours that touch it.
        let span = |d: i32| match d {
            -1 => 0..1,
            0 => 1..PADDED - 1,
            _ => PADDED - 1..PADDED,
        };
        for dy in -1..=1 {
            for dz in -1..=1 {
                for dx in -1..=1 {
                    if (dx, dy, dz) == (0, 0, 0) {
                        continue;
                    }
                    let Some(neighbour) =
                        world.chunk(ChunkPos(pos.0 + glam::IVec3::new(dx, dy, dz)))
                    else {
                        continue;
                    };
                    for y in span(dy) {
                        for z in span(dz) {
                            for x in span(dx) {
                                let [lx, ly, lz] =
                                    [x, y, z].map(|c| ((c + SIZE - 1) % SIZE) as u32);
                                let local = LocalPos::new(lx, ly, lz);
                                let index = padded_index([x, y, z]);
                                blocks[index] = neighbour.get(local);
                                light[index] = neighbour.light().get(local);
                            }
                        }
                    }
                }
            }
        }
        Some(Self { blocks, light })
    }

    fn get(&self, padded: [usize; 3]) -> BlockId {
        self.blocks[padded_index(padded)]
    }

    fn light(&self, padded: [usize; 3]) -> Light {
        self.light[padded_index(padded)]
    }
}

fn padded_index([x, y, z]: [usize; 3]) -> usize {
    (y * PADDED + z) * PADDED + x
}

/// The two axes spanning a face perpendicular to `axis`, as (u, v). For side
/// faces v is the vertical axis, so textures stand upright.
fn plane_axes(axis: usize) -> (usize, usize) {
    match axis {
        0 => (2, 1),
        1 => (0, 2),
        _ => (0, 1),
    }
}

/// A rectangle of identical faces, packed into four words for the GPU:
///
/// - word 0: x, y, z of its first block (5 bits each), width − 1 and
///   height − 1 along the face's u and v axes (5 bits each), face (3 bits),
///   whether to split it along the other diagonal (1 bit) and whether it
///   glows (1 bit);
/// - word 1: texture layer (10 bits), ambient occlusion of the four corners
///   (2 bits each), and 14 bits the renderer fills in;
/// - words 2 and 3: the light at the four corners, 16 bits each (see
///   [`Light`]).
///
/// Corner `i` is at u = `i & 1`, v = `i >> 1` of the face.
pub type Quad = [u32; 4];

/// Everything that has to match for neighbouring faces to merge.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct FaceLook {
    layer: u16,
    glows: bool,
    /// 0 (darkest) to 3 per corner.
    occlusion: [u8; 4],
    light: [Light; 4],
}

impl FaceLook {
    /// Packs into a non-zero cell value for [`merge`].
    fn to_cell(self) -> u128 {
        let mut cell = 1u128 << 127 | u128::from(self.layer) | u128::from(self.glows) << 16;
        for corner in 0..4 {
            cell |= u128::from(self.occlusion[corner]) << (17 + 2 * corner);
            cell |= u128::from(self.light[corner].to_raw()) << (32 + 16 * corner);
        }
        cell
    }

    fn from_cell(cell: u128) -> Self {
        Self {
            layer: cell as u16,
            glows: cell >> 16 & 1 != 0,
            occlusion: [0, 1, 2, 3].map(|corner| (cell >> (17 + 2 * corner) & 3) as u8),
            light: [0, 1, 2, 3].map(|corner| Light::from_raw((cell >> (32 + 16 * corner)) as u16)),
        }
    }

    /// How bright a corner looks, to pick the diagonal to split along.
    fn brightness(&self, corner: usize) -> u32 {
        let light = self.light[corner];
        u32::from(self.occlusion[corner]) * 64
            + (0..Light::CHANNELS)
                .map(|c| u32::from(light.channel(c)))
                .sum::<u32>()
    }
}

fn pack(block: [usize; 3], width: usize, height: usize, face: Face, look: FaceLook) -> Quad {
    let [x, y, z] = block.map(|c| c as u32);
    // Split along the diagonal whose corners are brighter, so a dark corner
    // fades evenly instead of drawing a dark line across the face.
    let flip = look.brightness(0) + look.brightness(3) > look.brightness(1) + look.brightness(2);
    let occlusion = look
        .occlusion
        .iter()
        .enumerate()
        .fold(0u32, |bits, (corner, &ao)| {
            bits | u32::from(ao) << (2 * corner)
        });
    let light = look.light.map(|light| u32::from(light.to_raw()));
    [
        x | y << 5
            | z << 10
            | (width as u32 - 1) << 15
            | (height as u32 - 1) << 20
            | (face.index() as u32) << 25
            | u32::from(flip) << 28
            | u32::from(look.glows) << 29,
        u32::from(look.layer) & 0x3ff | occlusion << 10,
        light[0] | light[1] << 16,
        light[2] | light[3] << 16,
    ]
}

/// How a face of the block at `padded` looks: the light of the open blocks
/// in front of each corner and how much the blocks around shade it.
fn face_look(
    chunk: &PaddedChunk,
    faces: &BlockFaces,
    padded: [usize; 3],
    face: Face,
    layer: u16,
) -> FaceLook {
    let axis = face.axis();
    let (u_axis, v_axis) = plane_axes(axis);
    let mut front = padded;
    front[axis] = if face.is_positive() {
        front[axis] + 1
    } else {
        front[axis] - 1
    };
    let step = |mut cell: [usize; 3], axis: usize, up: bool| {
        cell[axis] = if up { cell[axis] + 1 } else { cell[axis] - 1 };
        cell
    };
    let open = |cell: [usize; 3]| !faces.is_solid(chunk.get(cell));

    let mut occlusion = [0; 4];
    let mut light = [Light::DARK; 4];
    for corner in 0..4 {
        let (up_u, up_v) = (corner & 1 == 1, corner >> 1 == 1);
        let side_u = step(front, u_axis, up_u);
        let side_v = step(front, v_axis, up_v);
        let diagonal = step(side_u, v_axis, up_v);
        let (open_u, open_v) = (open(side_u), open(side_v));
        // Light can't squeeze between two blocks touching at an edge.
        let open_diagonal = (open_u || open_v) && open(diagonal);
        occlusion[corner] = if !open_u && !open_v {
            0
        } else {
            u8::from(open_u) + u8::from(open_v) + u8::from(open_diagonal)
        };

        let mut sum = [0u32; Light::CHANNELS];
        let mut count = 0;
        for (cell, open) in [
            (front, true),
            (side_u, open_u),
            (side_v, open_v),
            (diagonal, open_diagonal),
        ] {
            if open {
                let sample = chunk.light(cell);
                for (channel, sum) in sum.iter_mut().enumerate() {
                    *sum += u32::from(sample.channel(channel));
                }
                count += 1;
            }
        }
        light[corner] = Light::new(
            ((sum[0] * 2 + count) / (2 * count)) as u8,
            ((sum[1] * 2 + count) / (2 * count)) as u8,
            ((sum[2] * 2 + count) / (2 * count)) as u8,
            ((sum[3] * 2 + count) / (2 * count)) as u8,
        );
    }
    FaceLook {
        layer,
        glows: faces.glows(chunk.get(padded)),
        occlusion,
        light,
    }
}

/// The geometry of one chunk.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ChunkMesh {
    /// Grouped by the direction they face, in [`Face::ALL`] order.
    pub quads: Vec<Quad>,
    /// How many quads face each direction, in [`Face::ALL`] order.
    pub face_counts: [u32; 6],
    pub visibility: Visibility,
}

impl Default for ChunkMesh {
    fn default() -> Self {
        Self {
            quads: Vec::new(),
            face_counts: [0; 6],
            visibility: Visibility::ALL,
        }
    }
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}

/// Builds the chunk's visible faces, merging neighbouring faces that look
/// the same into larger rectangles (greedy meshing).
///
/// Visibility is found a whole column at a time: every column of the padded
/// chunk along each axis becomes a 64-bit mask of solid blocks, and
/// `column & !(column >> 1)` marks the blocks whose next neighbour is open.
pub fn mesh_chunk(chunk: &PaddedChunk, faces: &BlockFaces) -> ChunkMesh {
    // columns[axis][v * PADDED + u]: bit i is set if the block at padded
    // coordinate i along `axis` is solid.
    let mut columns = vec![[0u64; PADDED * PADDED]; 3];
    for (index, &block) in chunk.blocks.iter().enumerate() {
        if !faces.is_solid(block) {
            continue;
        }
        let p = [
            index % PADDED,
            index / (PADDED * PADDED),
            index / PADDED % PADDED,
        ];
        for (axis, columns) in columns.iter_mut().enumerate() {
            let (u_axis, v_axis) = plane_axes(axis);
            columns[p[v_axis] * PADDED + p[u_axis]] |= 1 << p[axis];
        }
    }

    let mut quads = Vec::new();
    let mut face_counts = [0; 6];
    for face in Face::ALL {
        let before = quads.len();
        let axis = face.axis();
        let (u_axis, v_axis) = plane_axes(axis);
        // visible[v][u]: bit s is set if the face of the block in slice s
        // at (u, v) is visible.
        let mut visible = [[0u32; SIZE]; SIZE];
        let mut slices_used = 0u32;
        for (v, row) in visible.iter_mut().enumerate() {
            for (u, visible) in row.iter_mut().enumerate() {
                let column = columns[axis][(v + 1) * PADDED + u + 1];
                let open = if face.is_positive() {
                    column & !(column >> 1)
                } else {
                    column & !(column << 1)
                };
                // Padded coordinates 1..=32 are the chunk's own slices 0..32.
                *visible = (open >> 1) as u32;
                slices_used |= *visible;
            }
        }
        while slices_used != 0 {
            let slice = slices_used.trailing_zeros() as usize;
            slices_used &= slices_used - 1;
            // Per face, how it looks; 0 where there is none.
            let mut plane = [[0u128; SIZE]; SIZE];
            for v in 0..SIZE {
                for u in 0..SIZE {
                    if visible[v][u] >> slice & 1 == 0 {
                        continue;
                    }
                    let mut padded = [0; 3];
                    padded[axis] = slice + 1;
                    padded[u_axis] = u + 1;
                    padded[v_axis] = v + 1;
                    let layers = faces.get(chunk.get(padded)).unwrap_or_default();
                    plane[v][u] =
                        face_look(chunk, faces, padded, face, layers[face.index()]).to_cell();
                }
            }
            merge(&mut plane, |u, v, width, height, cell| {
                let mut block = [0; 3];
                block[axis] = slice;
                block[u_axis] = u;
                block[v_axis] = v;
                quads.push(pack(block, width, height, face, FaceLook::from_cell(cell)));
            });
        }
        face_counts[face.index()] = (quads.len() - before) as u32;
    }

    // The chunk's own solid blocks as rows along x.
    let mut solid = [[0u32; SIZE]; SIZE];
    for (y, rows) in solid.iter_mut().enumerate() {
        for (z, row) in rows.iter_mut().enumerate() {
            *row = (columns[0][(y + 1) * PADDED + z + 1] >> 1) as u32;
        }
    }
    ChunkMesh {
        quads,
        face_counts,
        visibility: Visibility::of(&solid),
    }
}

/// Covers the non-zero cells with rectangles of equal cells, widest rows
/// first, and clears them.
fn merge(cells: &mut [[u128; SIZE]; SIZE], mut emit: impl FnMut(usize, usize, usize, usize, u128)) {
    for v in 0..SIZE {
        let mut u = 0;
        while u < SIZE {
            let cell = cells[v][u];
            if cell == 0 {
                u += 1;
                continue;
            }
            let mut width = 1;
            while u + width < SIZE && cells[v][u + width] == cell {
                width += 1;
            }
            let mut height = 1;
            while v + height < SIZE && cells[v + height][u..u + width].iter().all(|&c| c == cell) {
                height += 1;
            }
            for row in &mut cells[v..v + height] {
                row[u..u + width].fill(0);
            }
            emit(u, v, width, height, cell);
            u += width;
        }
    }
}

#[cfg(test)]
mod tests {
    use ruda_core::{BlockDef, ContentBuilder, CubeTextures};
    use ruda_world::Chunk;

    use super::*;

    struct Fixture {
        faces: BlockFaces,
        stone: BlockId,
        dirt: BlockId,
    }

    fn fixture() -> Fixture {
        let mut content = ContentBuilder::new();
        let mut add = |name: &str| {
            let id: ResourceId = format!("test:{name}").parse().unwrap();
            content
                .add_block(BlockDef::new(
                    id.clone(),
                    Appearance::Cube(CubeTextures::all(id)),
                ))
                .unwrap()
        };
        let (stone, dirt) = (add("stone"), add("dirt"));
        let content = content.build();
        let faces = BlockFaces::new(content.blocks(), |texture| {
            if texture.path() == "dirt" { 2 } else { 1 }
        });
        Fixture { faces, stone, dirt }
    }

    fn unpack(quad: Quad) -> ([u32; 3], u32, u32, Face, u32) {
        let w = quad[0];
        (
            [w & 31, (w >> 5) & 31, (w >> 10) & 31],
            ((w >> 15) & 31) + 1,
            ((w >> 20) & 31) + 1,
            Face::ALL[(w >> 25) as usize & 7],
            quad[1] & 0x3ff,
        )
    }

    /// A world with one chunk at the origin surrounded by air chunks.
    fn world_with(chunk: Chunk) -> World {
        let mut world = World::new();
        for y in -1..=1 {
            for z in -1..=1 {
                for x in -1..=1 {
                    world.insert_chunk(ChunkPos::new(x, y, z), Chunk::filled(BlockId::AIR));
                }
            }
        }
        world.insert_chunk(ChunkPos::new(0, 0, 0), chunk);
        world
    }

    #[test]
    fn a_lone_block_has_six_faces() {
        let f = fixture();
        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(3, 4, 5), f.stone);
        let padded = PaddedChunk::gather(&world_with(chunk), ChunkPos::new(0, 0, 0)).unwrap();
        let mesh = mesh_chunk(&padded, &f.faces);
        assert_eq!(mesh.quads.len(), 6);
        for quad in mesh.quads {
            let (block, width, height, _, layer) = unpack(quad);
            assert_eq!((block, width, height, layer), ([3, 4, 5], 1, 1, 1));
        }
    }

    #[test]
    fn a_floor_merges_into_one_quad_per_side() {
        let f = fixture();
        let mut chunk = Chunk::filled(BlockId::AIR);
        for x in 0..32 {
            for z in 0..32 {
                chunk.set(LocalPos::new(x, 0, z), f.stone);
            }
        }
        let padded = PaddedChunk::gather(&world_with(chunk), ChunkPos::new(0, 0, 0)).unwrap();
        let mesh = mesh_chunk(&padded, &f.faces);
        // Top and bottom are 32×32; the four sides are 32×1 strips.
        assert_eq!(mesh.quads.len(), 6);
        let top = mesh
            .quads
            .iter()
            .map(|&q| unpack(q))
            .find(|q| q.3 == Face::PosY)
            .unwrap();
        assert_eq!((top.1, top.2), (32, 32));
    }

    #[test]
    fn different_textures_do_not_merge() {
        let f = fixture();
        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(0, 0, 0), f.stone);
        chunk.set(LocalPos::new(1, 0, 0), f.dirt);
        let padded = PaddedChunk::gather(&world_with(chunk), ChunkPos::new(0, 0, 0)).unwrap();
        let tops: Vec<_> = mesh_chunk(&padded, &f.faces)
            .quads
            .into_iter()
            .map(unpack)
            .filter(|q| q.3 == Face::PosY)
            .collect();
        assert_eq!(tops.len(), 2);
    }

    #[test]
    fn covers_exactly_the_open_faces() {
        let f = fixture();
        let mut chunk = Chunk::filled(BlockId::AIR);
        // A deterministic jumble of two block types and air.
        let mut state = 12345u32;
        for pos in LocalPos::all() {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            match state >> 29 {
                0..=2 => {}
                3..=5 => drop(chunk.set(pos, f.stone)),
                _ => drop(chunk.set(pos, f.dirt)),
            }
        }
        let world = world_with(chunk.clone());
        let padded = PaddedChunk::gather(&world, ChunkPos::new(0, 0, 0)).unwrap();

        let solid = |pos: glam::IVec3| {
            let inside =
                pos.cmpge(glam::IVec3::ZERO).all() && pos.cmplt(glam::IVec3::splat(32)).all();
            inside
                && chunk.get(LocalPos::new(pos.x as u32, pos.y as u32, pos.z as u32))
                    != BlockId::AIR
        };
        let mut expected = [0u32; 6];
        for pos in LocalPos::all() {
            if !solid(pos.vec()) {
                continue;
            }
            for face in Face::ALL {
                if !solid(pos.vec() + face.normal()) {
                    expected[face.index()] += 1;
                }
            }
        }

        let mut covered = [0u32; 6];
        for quad in mesh_chunk(&padded, &f.faces).quads {
            let (_, width, height, face, _) = unpack(quad);
            covered[face.index()] += width * height;
        }
        assert_eq!(covered, expected);
    }

    #[test]
    fn groups_quads_by_face_and_finds_visibility() {
        let f = fixture();
        // A floor across the chunk at y = 4.
        let mut chunk = Chunk::filled(BlockId::AIR);
        for x in 0..32 {
            for z in 0..32 {
                chunk.set(LocalPos::new(x, 4, z), f.stone);
            }
        }
        chunk.set(LocalPos::new(3, 10, 3), f.dirt);
        let padded = PaddedChunk::gather(&world_with(chunk), ChunkPos::new(0, 0, 0)).unwrap();
        let mesh = mesh_chunk(&padded, &f.faces);
        assert_eq!(
            mesh.face_counts.iter().sum::<u32>() as usize,
            mesh.quads.len()
        );
        let mut start = 0;
        for face in Face::ALL {
            let count = mesh.face_counts[face.index()] as usize;
            assert!(
                mesh.quads[start..start + count]
                    .iter()
                    .all(|&q| unpack(q).3 == face)
            );
            start += count;
        }
        assert!(!mesh.visibility.connects(Face::NegY, Face::PosY));
        assert!(mesh.visibility.connects(Face::NegX, Face::PosX));
    }

    #[test]
    fn faces_towards_unloaded_neighbours_stay_hidden() {
        let f = fixture();
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 0, 0), Chunk::filled(f.stone));
        let padded = PaddedChunk::gather(&world, ChunkPos::new(0, 0, 0)).unwrap();
        assert!(mesh_chunk(&padded, &f.faces).is_empty());

        world.insert_chunk(ChunkPos::new(0, 1, 0), Chunk::filled(BlockId::AIR));
        let padded = PaddedChunk::gather(&world, ChunkPos::new(0, 0, 0)).unwrap();
        let mesh = mesh_chunk(&padded, &f.faces);
        // Only the top shows; its edges are shaded by the missing neighbours.
        let area: u32 = mesh
            .quads
            .iter()
            .map(|&quad| unpack(quad))
            .inspect(|quad| assert_eq!(quad.3, Face::PosY))
            .map(|(_, width, height, _, _)| width * height)
            .sum();
        assert_eq!(area, 32 * 32);
    }
}
