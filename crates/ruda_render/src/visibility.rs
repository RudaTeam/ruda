//! Which faces of a chunk can see each other through its open blocks, for
//! skipping chunks hidden behind solid ground (cave culling).

use ruda_core::Face;

const SIZE: usize = ruda_core::CHUNK_SIZE as usize;

/// For each face of a chunk, the faces reachable from it through open
/// blocks inside the chunk.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Visibility([u8; 6]);

impl Visibility {
    /// Nothing to see through: a chunk of solid blocks.
    pub const NONE: Self = Self([0; 6]);
    /// Open from every face to every other: a chunk of air.
    pub const ALL: Self = Self([0b11_1111; 6]);

    /// For each face in [`Face::ALL`] order, a bit mask of the faces it reaches.
    #[cfg(test)]
    pub(crate) fn from_faces(faces: [u8; 6]) -> Self {
        Self(faces)
    }

    /// Whether something entering through `from` can leave through `to`.
    pub fn connects(self, from: Face, to: Face) -> bool {
        self.0[from.index()] & (1 << to.index()) != 0
    }

    /// From the chunk's solid blocks: `solid[y][z]` has bit x set for a solid
    /// block at (x, y, z).
    pub fn of(solid: &[[u32; SIZE]; SIZE]) -> Self {
        let mut visited = [[0u32; SIZE]; SIZE];
        let mut connections = [0u8; 6];
        let mut stack = Vec::new();
        for y in 0..SIZE {
            for z in 0..SIZE {
                loop {
                    let unvisited = !solid[y][z] & !visited[y][z];
                    if unvisited == 0 {
                        break;
                    }
                    // Flood one open region from its lowest unvisited block,
                    // a row segment at a time.
                    let mut faces = 0u8;
                    stack.push((y, z, unvisited.isolate_lowest_one()));
                    while let Some((y, z, seed)) = stack.pop() {
                        let open = !solid[y][z] & !visited[y][z];
                        let row = fill(seed & open, open);
                        if row == 0 {
                            continue;
                        }
                        visited[y][z] |= row;
                        faces |= touched_faces(row, y, z);
                        for (ny, nz) in neighbours(y, z) {
                            let next = row & !solid[ny][nz] & !visited[ny][nz];
                            if next != 0 {
                                stack.push((ny, nz, next));
                            }
                        }
                    }
                    for face in Face::ALL {
                        if faces & (1 << face.index()) != 0 {
                            connections[face.index()] |= faces;
                        }
                    }
                }
            }
        }
        Self(connections)
    }
}

/// The runs of `open` bits that contain a bit of `seed`.
fn fill(seed: u32, open: u32) -> u32 {
    // Kogge–Stone fill towards higher bits, then towards lower ones.
    let up = |mut fill: u32| {
        let mut through = open;
        for shift in [1, 2, 4, 8, 16] {
            fill |= through & (fill << shift);
            through &= through << shift;
        }
        fill
    };
    let down = |mut fill: u32| {
        let mut through = open;
        for shift in [1, 2, 4, 8, 16] {
            fill |= through & (fill >> shift);
            through &= through >> shift;
        }
        fill
    };
    up(seed) | down(seed)
}

/// Faces of the chunk that the open blocks `row` of row (y, z) lie on.
fn touched_faces(row: u32, y: usize, z: usize) -> u8 {
    let mut faces = 0u8;
    let mut touch = |face: Face, touched: bool| {
        if touched {
            faces |= 1 << face.index();
        }
    };
    touch(Face::NegX, row & 1 != 0);
    touch(Face::PosX, row & (1 << (SIZE - 1)) != 0);
    touch(Face::NegY, y == 0);
    touch(Face::PosY, y == SIZE - 1);
    touch(Face::NegZ, z == 0);
    touch(Face::PosZ, z == SIZE - 1);
    faces
}

fn neighbours(y: usize, z: usize) -> impl Iterator<Item = (usize, usize)> {
    [
        (y.wrapping_sub(1), z),
        (y + 1, z),
        (y, z.wrapping_sub(1)),
        (y, z + 1),
    ]
    .into_iter()
    .filter(|&(y, z)| y < SIZE && z < SIZE)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn air_and_stone() {
        assert_eq!(Visibility::of(&[[0; SIZE]; SIZE]), Visibility::ALL);
        assert_eq!(Visibility::of(&[[u32::MAX; SIZE]; SIZE]), Visibility::NONE);
    }

    #[test]
    fn a_wall_splits_the_chunk() {
        // Solid at x = 10: the −X side can't see the +X side.
        let solid = [[1 << 10; SIZE]; SIZE];
        let visibility = Visibility::of(&solid);
        assert!(!visibility.connects(Face::NegX, Face::PosX));
        assert!(visibility.connects(Face::NegX, Face::PosY));
        assert!(visibility.connects(Face::PosZ, Face::PosX));
        assert!(visibility.connects(Face::NegY, Face::PosY));
    }

    #[test]
    fn a_winding_tunnel_connects_its_ends() {
        // Solid everywhere except a tunnel entering at x = 0, y = 5, z = 3,
        // running along x to 20, then up along y to the top.
        let mut solid = [[u32::MAX; SIZE]; SIZE];
        solid[5][3] &= !((1 << 21) - 1);
        for row in &mut solid[5..] {
            row[3] &= !(1 << 20);
        }
        let visibility = Visibility::of(&solid);
        assert!(visibility.connects(Face::NegX, Face::PosY));
        assert!(visibility.connects(Face::PosY, Face::NegX));
        assert!(!visibility.connects(Face::NegX, Face::PosX));
        assert!(!visibility.connects(Face::NegZ, Face::PosZ));
    }

    #[test]
    fn fills_whole_runs() {
        assert_eq!(fill(1 << 4, 0b0111_1100), 0b0111_1100);
        assert_eq!(fill(1 << 4, 0b1100_0111_0000), 0b0111_0000);
        assert_eq!(fill(0, u32::MAX), 0);
        assert_eq!(fill(1 << 31, u32::MAX), u32::MAX);
    }
}
