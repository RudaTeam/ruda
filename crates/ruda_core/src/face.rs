use glam::IVec3;

/// One of the six faces of a block, named after the direction it faces.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Face {
    PosX,
    NegX,
    PosY,
    NegY,
    PosZ,
    NegZ,
}

impl Face {
    pub const ALL: [Face; 6] = [
        Face::PosX,
        Face::NegX,
        Face::PosY,
        Face::NegY,
        Face::PosZ,
        Face::NegZ,
    ];

    /// Position in [`Face::ALL`]; fits in three bits.
    pub fn index(self) -> usize {
        self as usize
    }

    /// 0 for x, 1 for y, 2 for z.
    pub fn axis(self) -> usize {
        self.index() / 2
    }

    pub fn is_positive(self) -> bool {
        self.index().is_multiple_of(2)
    }

    pub fn normal(self) -> IVec3 {
        let sign = if self.is_positive() { 1 } else { -1 };
        let mut normal = IVec3::ZERO;
        normal[self.axis()] = sign;
        normal
    }

    pub fn opposite(self) -> Face {
        Face::ALL[self.index() ^ 1]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normals_and_opposites_agree() {
        for face in Face::ALL {
            assert_eq!(face.opposite().normal(), -face.normal());
            assert_eq!(face.opposite().opposite(), face);
            assert_eq!(face.normal()[face.axis()].abs(), 1);
        }
        assert_eq!(Face::PosY.normal(), IVec3::Y);
        assert_eq!(Face::NegZ.normal(), IVec3::NEG_Z);
    }
}
