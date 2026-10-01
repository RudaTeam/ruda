use ruda_core::CHUNK_VOLUME;

/// One value per block of a chunk, stored compactly.
///
/// Most chunks hold few distinct values, so instead of the values themselves
/// the palette keeps a list of the distinct ones plus a small index into it
/// for every block. A chunk where every block has the same value, like open
/// sky or deep stone, takes no per-block memory at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Palette<T> {
    Uniform(T),
    Paletted(Paletted<T>),
}

/// Bits per palette index: powers of two, so an index never straddles two words.
const INDEX_BITS: [u32; 6] = [1, 2, 4, 8, 16, 32];

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Paletted<T> {
    palette: Vec<T>,
    bits: u32,
    words: Box<[u64]>,
}

impl<T: Copy + Eq> Palette<T> {
    pub fn filled(value: T) -> Self {
        Self::Uniform(value)
    }

    /// One value per block, in index order.
    pub fn from_slice(values: &[T]) -> Self {
        assert_eq!(
            values.len(),
            CHUNK_VOLUME,
            "a chunk has {CHUNK_VOLUME} blocks"
        );
        let mut palette = vec![values[0]];
        let mut indices = Vec::with_capacity(CHUNK_VOLUME);
        // Runs of the same value are common; only look it up when it changes.
        let mut last = (values[0], 0);
        for &value in values {
            if value != last.0 {
                let index = match palette.iter().position(|&known| known == value) {
                    Some(index) => index,
                    None => {
                        palette.push(value);
                        palette.len() - 1
                    }
                };
                last = (value, index);
            }
            indices.push(last.1);
        }
        if palette.len() == 1 {
            return Self::Uniform(values[0]);
        }
        let bits = INDEX_BITS
            .into_iter()
            .find(|&bits| palette.len() <= 1 << bits)
            .expect("32-bit indices fit any palette");
        let mut paletted = Paletted {
            palette,
            bits,
            words: vec![0; words_for(bits)].into(),
        };
        for (at, index) in indices.into_iter().enumerate() {
            paletted.set_index_at(at, index);
        }
        Self::Paletted(paletted)
    }

    pub fn get(&self, index: usize) -> T {
        match self {
            Self::Uniform(value) => *value,
            Self::Paletted(paletted) => paletted.get(index),
        }
    }

    /// Sets a value and returns the one it replaced.
    pub fn set(&mut self, index: usize, value: T) -> T {
        match self {
            Self::Uniform(current) if *current == value => value,
            Self::Uniform(current) => {
                let previous = *current;
                let mut paletted = Paletted::filled(previous);
                paletted.set(index, value);
                *self = Self::Paletted(paletted);
                previous
            }
            Self::Paletted(paletted) => paletted.set(index, value),
        }
    }

    /// The value of every block, if they are all the same.
    pub fn uniform(&self) -> Option<T> {
        match self {
            Self::Uniform(value) => Some(*value),
            Self::Paletted(_) => None,
        }
    }

    /// The distinct values; may include some no block has any more.
    pub fn palette(&self) -> &[T] {
        match self {
            Self::Uniform(value) => std::slice::from_ref(value),
            Self::Paletted(paletted) => &paletted.palette,
        }
    }

    /// Writes every value into `out`, in index order.
    pub fn copy_to(&self, out: &mut [T]) {
        assert_eq!(out.len(), CHUNK_VOLUME, "a chunk has {CHUNK_VOLUME} blocks");
        match self {
            Self::Uniform(value) => out.fill(*value),
            Self::Paletted(paletted) => paletted.copy_to(out),
        }
    }
}

impl<T: Copy + Eq> Paletted<T> {
    /// One-bit indices, all pointing at `value`.
    fn filled(value: T) -> Self {
        Self {
            palette: vec![value],
            bits: INDEX_BITS[0],
            words: vec![0; words_for(INDEX_BITS[0])].into(),
        }
    }

    fn get(&self, index: usize) -> T {
        self.palette[self.index_at(index)]
    }

    fn set(&mut self, index: usize, value: T) -> T {
        let previous = self.get(index);
        if previous != value {
            let palette_index = self.palette_index(value);
            self.set_index_at(index, palette_index);
        }
        previous
    }

    fn copy_to(&self, out: &mut [T]) {
        let per_word = 64 / self.bits as usize;
        let mask = index_mask(self.bits);
        for (word, values) in self.words.iter().zip(out.chunks_mut(per_word)) {
            for (slot, value) in values.iter_mut().enumerate() {
                let palette_index = (word >> (slot as u32 * self.bits)) & mask;
                *value = self.palette[palette_index as usize];
            }
        }
    }

    /// Position of `value` in the palette, adding it (and widening the
    /// indices when they run out of bits) if it is not there yet.
    fn palette_index(&mut self, value: T) -> usize {
        if let Some(position) = self.palette.iter().position(|&known| known == value) {
            return position;
        }
        self.palette.push(value);
        if self.palette.len() > 1 << self.bits {
            self.widen();
        }
        self.palette.len() - 1
    }

    fn widen(&mut self) {
        let bits = INDEX_BITS
            .into_iter()
            .find(|&bits| bits > self.bits)
            .expect("32-bit indices fit any palette");
        let mut widened = Self {
            palette: Vec::new(),
            bits,
            words: vec![0; words_for(bits)].into(),
        };
        for index in 0..CHUNK_VOLUME {
            widened.set_index_at(index, self.index_at(index));
        }
        self.bits = bits;
        self.words = widened.words;
    }

    fn index_at(&self, index: usize) -> usize {
        let per_word = 64 / self.bits as usize;
        let shift = (index % per_word) as u32 * self.bits;
        ((self.words[index / per_word] >> shift) & index_mask(self.bits)) as usize
    }

    fn set_index_at(&mut self, index: usize, palette_index: usize) {
        let per_word = 64 / self.bits as usize;
        let shift = (index % per_word) as u32 * self.bits;
        let mask = index_mask(self.bits);
        let word = &mut self.words[index / per_word];
        *word = (*word & !(mask << shift)) | ((palette_index as u64 & mask) << shift);
    }
}

/// A palette travels as its distinct values plus the packed indices as
/// little-endian bytes, and is validated in full when it arrives.
#[cfg(feature = "serde")]
mod wire {
    use serde::{Deserialize, Deserializer, Serialize, Serializer, de::Error as _};

    use super::*;

    #[derive(Serialize, Deserialize)]
    enum Wire<T> {
        Uniform(T),
        Paletted {
            palette: Vec<T>,
            bits: u32,
            indices: Vec<u8>,
        },
    }

    impl<T: Copy + Eq + Serialize> Serialize for Palette<T> {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            let wire = match self {
                Palette::Uniform(value) => Wire::Uniform(*value),
                Palette::Paletted(paletted) => Wire::Paletted {
                    palette: paletted.palette.clone(),
                    bits: paletted.bits,
                    indices: paletted
                        .words
                        .iter()
                        .flat_map(|w| w.to_le_bytes())
                        .collect(),
                },
            };
            wire.serialize(serializer)
        }
    }

    impl<'de, T: Copy + Eq + Deserialize<'de>> Deserialize<'de> for Palette<T> {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            Ok(match Wire::<T>::deserialize(deserializer)? {
                Wire::Uniform(value) => Palette::Uniform(value),
                Wire::Paletted {
                    palette,
                    bits,
                    indices,
                } => {
                    Palette::Paletted(paletted(palette, bits, &indices).map_err(D::Error::custom)?)
                }
            })
        }
    }

    fn paletted<T: Copy + Eq>(
        palette: Vec<T>,
        bits: u32,
        indices: &[u8],
    ) -> Result<Paletted<T>, &'static str> {
        if !INDEX_BITS.contains(&bits) {
            return Err("unsupported palette index width");
        }
        if palette.is_empty() || palette.len() as u64 > 1u64 << bits {
            return Err("palette does not match its index width");
        }
        if indices.len() != words_for(bits) * 8 {
            return Err("wrong amount of palette index data");
        }
        let (words, _) = indices.as_chunks::<8>();
        let words = words.iter().copied().map(u64::from_le_bytes).collect();
        let paletted = Paletted {
            palette,
            bits,
            words,
        };
        if (0..CHUNK_VOLUME).any(|index| paletted.index_at(index) >= paletted.palette.len()) {
            return Err("palette index out of range");
        }
        Ok(paletted)
    }
}

fn words_for(bits: u32) -> usize {
    CHUNK_VOLUME * bits as usize / 64
}

fn index_mask(bits: u32) -> u64 {
    (1u64 << bits) - 1
}

#[cfg(test)]
mod tests {
    use ruda_core::{BlockId, LocalPos};

    use super::*;

    fn id(raw: u32) -> BlockId {
        BlockId::from_raw(raw)
    }

    #[test]
    fn stays_uniform_until_a_different_value_is_set() {
        let mut palette = Palette::filled(id(5));
        assert_eq!(palette.set(100, id(5)), id(5));
        assert_eq!(palette.uniform(), Some(id(5)));

        assert_eq!(palette.set(100, id(7)), id(5));
        assert_eq!(palette.uniform(), None);
        assert_eq!(palette.get(100), id(7));
        assert_eq!(palette.get(0), id(5));
        assert_eq!(palette.palette(), [id(5), id(7)]);
    }

    #[test]
    fn widens_indices_as_the_palette_grows() {
        let mut palette = Palette::filled(BlockId::AIR);
        // 300 distinct values need 16-bit indices.
        for (n, pos) in LocalPos::all().step_by(97).take(300).enumerate() {
            palette.set(pos.index(), id(n as u32 + 10));
        }
        for (n, pos) in LocalPos::all().step_by(97).take(300).enumerate() {
            assert_eq!(palette.get(pos.index()), id(n as u32 + 10));
        }
        assert_eq!(palette.get(CHUNK_VOLUME - 1), BlockId::AIR);
    }

    #[test]
    fn from_slice_and_copy_to_round_trip() {
        let values: Vec<BlockId> = (0..CHUNK_VOLUME)
            .map(|i| id((i * 7919 % 13) as u32))
            .collect();
        let palette = Palette::from_slice(&values);
        let mut copy = vec![BlockId::AIR; CHUNK_VOLUME];
        palette.copy_to(&mut copy);
        assert_eq!(copy, values);
    }
}
