//! Turning chunks into geometry: one packed quad per visible run of faces,
//! with the light and ambient occlusion at each of its corners.

use ruda_core::{
    Appearance, BlockId, BlockRegistry, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, Face, Light, LocalPos,
    Mount, ResourceId,
};
use ruda_world::World;

use crate::Visibility;

const SIZE: usize = CHUNK_SIZE as usize;
/// A chunk plus a one-block border on every side.
const PADDED: usize = SIZE + 2;

/// For every block, the texture layer of each face, or `None` for blocks
/// that are not cubes, whether it glows, and the model of blocks that
/// aren't cubes.
#[derive(Clone, Debug, Default)]
pub struct BlockFaces {
    faces: Vec<Option<[u16; 6]>>,
    glowing: Vec<bool>,
    models: Vec<Option<Model>>,
}

/// A block drawn from triangles rather than as a cube.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Model {
    Torch { layer: u16, mount: Mount },
}

impl BlockFaces {
    /// `layer` maps a texture to its layer in the block texture array.
    pub fn new(blocks: &BlockRegistry, layer: impl Fn(&ResourceId) -> u16) -> Self {
        let faces = blocks
            .iter()
            .map(|(_, def)| match &def.appearance {
                Appearance::Cube(textures) => {
                    Some(Face::ALL.map(|face| layer(textures.for_face(face))))
                }
                _ => None,
            })
            .collect();
        let glowing = blocks.iter().map(|(_, def)| !def.light.is_dark()).collect();
        let models = blocks
            .iter()
            .map(|(id, def)| match &def.appearance {
                Appearance::Torch { texture } => Some(Model::Torch {
                    layer: layer(texture),
                    mount: blocks.mount(id)?,
                }),
                _ => None,
            })
            .collect();
        Self {
            faces,
            glowing,
            models,
        }
    }

    fn model(&self, id: BlockId) -> Option<Model> {
        self.models.get(id.index()).copied().flatten()
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

    /// The texture layer of each face of a cube block.
    pub(crate) fn layers(&self, id: BlockId) -> Option<[u16; 6]> {
        self.get(id)
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

/// A vertex of a block model, packed into four words like a [`Quad`]:
///
/// - word 0: x, y, z relative to the chunk in sixteenths of a block, plus
///   256 (10 bits each), and whether it glows (1 bit);
/// - word 1: texture u and v in pixels (5 bits each), texture layer (8 bits),
///   and 14 bits the renderer fills in;
/// - word 2: light (16 bits) and the face it shades like (3 bits).
pub type ModelVertex = [u32; 4];

/// The corners of the faces of a box, counter-clockwise seen from outside;
/// bits 0, 1, 2 of each are x, y, z.
const BOX_FACES: [(Face, [u8; 4]); 6] = [
    (Face::PosX, [0b101, 0b001, 0b011, 0b111]),
    (Face::NegX, [0b000, 0b100, 0b110, 0b010]),
    (Face::PosY, [0b110, 0b111, 0b011, 0b010]),
    (Face::NegY, [0b000, 0b001, 0b101, 0b100]),
    (Face::PosZ, [0b100, 0b101, 0b111, 0b110]),
    (Face::NegZ, [0b001, 0b000, 0b010, 0b011]),
];

/// A torch: a 2×10×2-pixel stick. On a wall it starts at the wall and its
/// top leans 4 pixels out. Positions are in sixteenths of a block.
fn torch_vertices(
    block: [u32; 3],
    mount: Mount,
    layer: u16,
    light: Light,
    out: &mut Vec<ModelVertex>,
) {
    let corner = |bits: u8| -> [i32; 3] {
        let [x, y, z] = [bits & 1, bits >> 1 & 1, bits >> 2 & 1].map(i32::from);
        match mount {
            Mount::Floor => [7 + 2 * x, 10 * y, 7 + 2 * z],
            Mount::Wall(facing) => {
                let out_axis = facing.axis();
                let sign = if facing.is_positive() { 1 } else { -1 };
                // Middle of the stick along the way it leans, bottom to top.
                let middle = if facing.is_positive() { 1 } else { 15 } + 4 * sign * y;
                let mut position = [7 + 2 * x, 3 + 10 * y, 7 + 2 * z];
                let along = [x, y, z][out_axis];
                position[out_axis] = middle + 2 * along - 1;
                position
            }
        }
    };
    let origin = block.map(|c| c as i32 * 16);
    for (face, corners) in BOX_FACES {
        // Pixels of the texture: the stick's sides, its flame or its end.
        let (top, bottom) = match face {
            Face::PosY => (6, 8),
            Face::NegY => (14, 16),
            _ => (6, 16),
        };
        let uv = [(7, bottom), (9, bottom), (9, top), (7, top)];
        for index in [0, 1, 2, 0, 2, 3] {
            let p = corner(corners[index]);
            let [x, y, z] = [0, 1, 2].map(|axis| (origin[axis] + p[axis] + 256) as u32);
            let (u, v) = uv[index];
            out.push([
                x | y << 10 | z << 20 | 1 << 30,
                u | v << 5 | u32::from(layer & 0xff) << 10,
                u32::from(light.to_raw()) | (face.index() as u32) << 16,
                0,
            ]);
        }
    }
}

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
    /// Triangles of the blocks that aren't cubes.
    pub models: Vec<ModelVertex>,
    /// How many quads face each direction, in [`Face::ALL`] order.
    pub face_counts: [u32; 6],
    pub visibility: Visibility,
    /// Which blocks are solid cubes, for rays towards the sun: a column of
    /// 32 bits up y for each x and z, entry `z * 32 + x`; `None` if none is.
    pub solids: Option<Box<[u32; SIZE * SIZE]>>,
}

impl Default for ChunkMesh {
    fn default() -> Self {
        Self {
            quads: Vec::new(),
            models: Vec::new(),
            face_counts: [0; 6],
            visibility: Visibility::ALL,
            solids: None,
        }
    }
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty() && self.models.is_empty()
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

    let mut models = Vec::new();
    for index in 0..CHUNK_VOLUME {
        let local = LocalPos::from_index(index);
        let padded = [local.x(), local.y(), local.z()].map(|c| c as usize + 1);
        if let Some(Model::Torch { layer, mount }) = faces.model(chunk.get(padded)) {
            let block = [local.x(), local.y(), local.z()];
            torch_vertices(block, mount, layer, chunk.light(padded), &mut models);
        }
    }

    // The chunk's own solid blocks as rows along x.
    let mut solid = [[0u32; SIZE]; SIZE];
    for (y, rows) in solid.iter_mut().enumerate() {
        for (z, row) in rows.iter_mut().enumerate() {
            *row = (columns[0][(y + 1) * PADDED + z + 1] >> 1) as u32;
        }
    }
    // And as columns up y.
    let mut solids = Box::new([0u32; SIZE * SIZE]);
    for (z, row) in solids.chunks_mut(SIZE).enumerate() {
        for (x, column) in row.iter_mut().enumerate() {
            *column = (columns[1][(z + 1) * PADDED + x + 1] >> 1) as u32;
        }
    }
    ChunkMesh {
        quads,
        models,
        face_counts,
        visibility: Visibility::of(&solid),
        solids: solids.iter().any(|&column| column != 0).then_some(solids),
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
    fn solids_mark_the_cubes_in_columns() {
        let f = fixture();
        assert!(
            mesh_chunk(
                &PaddedChunk::gather(
                    &world_with(Chunk::filled(BlockId::AIR)),
                    ChunkPos::new(0, 0, 0)
                )
                .unwrap(),
                &f.faces
            )
            .solids
            .is_none()
        );
        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(3, 4, 5), f.stone);
        chunk.set(LocalPos::new(3, 31, 5), f.dirt);
        let padded = PaddedChunk::gather(&world_with(chunk), ChunkPos::new(0, 0, 0)).unwrap();
        let solids = mesh_chunk(&padded, &f.faces).solids.unwrap();
        assert_eq!(solids[5 * 32 + 3], 1 << 4 | 1 << 31);
        assert_eq!(solids.iter().filter(|&&column| column != 0).count(), 1);
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
    fn torches_are_models_on_the_floor_or_leaning_off_a_wall() {
        let mut content = ContentBuilder::new();
        let id: ResourceId = "test:torch".parse().unwrap();
        let torch = content
            .add_block(BlockDef::new(id.clone(), Appearance::Torch { texture: id }))
            .unwrap();
        let content = content.build();
        let blocks = content.blocks();
        let faces = BlockFaces::new(blocks, |_| 3);

        // Sixteenths of a block within the block at local (x, y, z).
        let positions = |mount_face: Face| {
            let mut chunk = Chunk::filled(BlockId::AIR);
            chunk.set(
                LocalPos::new(4, 5, 6),
                blocks.placed(torch, mount_face).unwrap(),
            );
            let padded = PaddedChunk::gather(&world_with(chunk), ChunkPos::new(0, 0, 0)).unwrap();
            let mesh = mesh_chunk(&padded, &faces);
            assert!(mesh.quads.is_empty());
            assert_eq!(mesh.models.len(), 36);
            mesh.models
                .iter()
                .map(|v| {
                    let [x, y, z] = [0, 10, 20].map(|shift| ((v[0] >> shift) & 1023) as i32 - 256);
                    [x - 64, y - 80, z - 96]
                })
                .collect::<Vec<_>>()
        };
        let range = |points: &[[i32; 3]], axis: usize| {
            let values = points.iter().map(|p| p[axis]);
            (values.clone().min().unwrap(), values.max().unwrap())
        };

        let floor = positions(Face::PosY);
        assert_eq!(range(&floor, 0), (7, 9));
        assert_eq!(range(&floor, 1), (0, 10));

        // Hanging on the block at +z, leaning towards −z.
        let wall = positions(Face::NegZ);
        assert_eq!(range(&wall, 2), (10, 16));
        assert_eq!(range(&wall, 1), (3, 13));
        let bottom: Vec<[i32; 3]> = wall.iter().copied().filter(|p| p[1] == 3).collect();
        assert_eq!(range(&bottom, 2), (14, 16));
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
