//! Turning chunks into geometry: one packed quad per visible run of faces.

use ruda_core::{
    Appearance, BlockId, BlockRegistry, CHUNK_SIZE, CHUNK_VOLUME, ChunkPos, Face, LocalPos,
    ResourceId,
};
use ruda_world::World;

const SIZE: usize = CHUNK_SIZE as usize;
/// A chunk plus a one-block border on every side.
const PADDED: usize = SIZE + 2;

/// For every block, the texture layer of each face, or `None` for blocks
/// that are not drawn.
#[derive(Clone, Debug, Default)]
pub struct BlockFaces {
    faces: Vec<Option<[u16; 6]>>,
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
        Self { faces }
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
}

/// A chunk's blocks with a one-block border copied from its six neighbours,
/// enough to tell which faces touch air.
#[derive(Clone, Debug)]
pub struct PaddedChunk {
    blocks: Box<[BlockId]>,
}

impl PaddedChunk {
    /// `None` if the chunk itself is not loaded. Neighbours that are not
    /// loaded count as solid, so faces towards them stay hidden until they
    /// arrive and the chunk is meshed again.
    pub fn gather(world: &World, pos: ChunkPos) -> Option<Self> {
        let chunk = world.chunk(pos)?;
        let mut inner = vec![BlockId::AIR; CHUNK_VOLUME];
        chunk.copy_to(&mut inner);
        let mut blocks = vec![BlockId::AIR; PADDED * PADDED * PADDED].into_boxed_slice();
        for (index, &block) in inner.iter().enumerate() {
            let local = LocalPos::from_index(index);
            blocks[padded_index([local.x(), local.y(), local.z()].map(|c| c as usize + 1))] = block;
        }

        for face in Face::ALL {
            let neighbour = world.chunk(pos.offset(face));
            let axis = face.axis();
            let (u_axis, v_axis) = plane_axes(axis);
            // The neighbour's layer that touches this chunk, and where it goes
            // in the border.
            let (source, border) = if face.is_positive() {
                (0, PADDED - 1)
            } else {
                (SIZE - 1, 0)
            };
            for v in 0..SIZE {
                for u in 0..SIZE {
                    let mut local = [0; 3];
                    local[axis] = source;
                    local[u_axis] = u;
                    local[v_axis] = v;
                    let block = neighbour.map_or(BlockId::UNKNOWN, |chunk| {
                        chunk.get(LocalPos::new(
                            local[0] as u32,
                            local[1] as u32,
                            local[2] as u32,
                        ))
                    });
                    let mut padded = [0; 3];
                    padded[axis] = border;
                    padded[u_axis] = u + 1;
                    padded[v_axis] = v + 1;
                    blocks[padded_index(padded)] = block;
                }
            }
        }
        Some(Self { blocks })
    }

    fn get(&self, padded: [usize; 3]) -> BlockId {
        self.blocks[padded_index(padded)]
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

/// A rectangle of identical faces, packed into two words for the GPU:
///
/// - word 0: x, y, z of its first block (5 bits each), width − 1 and
///   height − 1 along the face's u and v axes (5 bits each), face (3 bits);
/// - word 1: texture layer (16 bits).
pub type Quad = [u32; 2];

fn pack(block: [usize; 3], width: usize, height: usize, face: Face, layer: u16) -> Quad {
    let [x, y, z] = block.map(|c| c as u32);
    [
        x | y << 5
            | z << 10
            | (width as u32 - 1) << 15
            | (height as u32 - 1) << 20
            | (face.index() as u32) << 25,
        u32::from(layer),
    ]
}

/// The geometry of one chunk.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct ChunkMesh {
    pub quads: Vec<Quad>,
}

impl ChunkMesh {
    pub fn is_empty(&self) -> bool {
        self.quads.is_empty()
    }
}

/// Builds the chunk's visible faces, merging neighbouring faces with the
/// same texture into larger rectangles (greedy meshing).
pub fn mesh_chunk(chunk: &PaddedChunk, faces: &BlockFaces) -> ChunkMesh {
    let mut quads = Vec::new();
    // Texture layer + 1 of each visible face in a slice; 0 where there is none.
    let mut cells = [[0u32; SIZE]; SIZE];
    for face in Face::ALL {
        let axis = face.axis();
        let (u_axis, v_axis) = plane_axes(axis);
        let normal = face.normal();
        for slice in 0..SIZE {
            let mut any = false;
            for (v, row) in cells.iter_mut().enumerate() {
                for (u, cell) in row.iter_mut().enumerate() {
                    let mut block = [0; 3];
                    block[axis] = slice;
                    block[u_axis] = u;
                    block[v_axis] = v;
                    let padded = block.map(|c| c + 1);
                    *cell = 0;
                    let Some(layers) = faces.get(chunk.get(padded)) else {
                        continue;
                    };
                    let neighbour =
                        std::array::from_fn(|i| (padded[i] as i32 + normal[i]) as usize);
                    if faces.is_solid(chunk.get(neighbour)) {
                        continue;
                    }
                    *cell = u32::from(layers[face.index()]) + 1;
                    any = true;
                }
            }
            if any {
                merge(&mut cells, |u, v, width, height, cell| {
                    let mut block = [0; 3];
                    block[axis] = slice;
                    block[u_axis] = u;
                    block[v_axis] = v;
                    quads.push(pack(block, width, height, face, (cell - 1) as u16));
                });
            }
        }
    }
    ChunkMesh { quads }
}

/// Covers the non-zero cells with rectangles of equal cells, widest rows
/// first, and clears them.
fn merge(cells: &mut [[u32; SIZE]; SIZE], mut emit: impl FnMut(usize, usize, usize, usize, u32)) {
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
            quad[1],
        )
    }

    /// A world with one chunk at the origin surrounded by air chunks.
    fn world_with(chunk: Chunk) -> World {
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 0, 0), chunk);
        for face in Face::ALL {
            world.insert_chunk(
                ChunkPos::new(0, 0, 0).offset(face),
                Chunk::filled(BlockId::AIR),
            );
        }
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
    fn faces_towards_unloaded_neighbours_stay_hidden() {
        let f = fixture();
        let mut world = World::new();
        world.insert_chunk(ChunkPos::new(0, 0, 0), Chunk::filled(f.stone));
        let padded = PaddedChunk::gather(&world, ChunkPos::new(0, 0, 0)).unwrap();
        assert!(mesh_chunk(&padded, &f.faces).is_empty());

        world.insert_chunk(ChunkPos::new(0, 1, 0), Chunk::filled(BlockId::AIR));
        let padded = PaddedChunk::gather(&world, ChunkPos::new(0, 0, 0)).unwrap();
        let mesh = mesh_chunk(&padded, &f.faces);
        assert_eq!(mesh.quads.len(), 1);
        assert_eq!(unpack(mesh.quads[0]).3, Face::PosY);
    }
}
