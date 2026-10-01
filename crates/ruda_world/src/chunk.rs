use ruda_core::{BlockId, CHUNK_VOLUME, LocalPos};

/// A 32×32×32 cube of blocks.
///
/// Most chunks hold very few kinds of blocks, so instead of one id per block
/// a chunk keeps a palette of the ids it contains plus a small index into it
/// for every block. A chunk made of a single block, like open sky or deep
/// stone, takes no per-block memory at all.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Chunk {
    storage: Storage,
}

#[derive(Clone, Debug, PartialEq, Eq)]
enum Storage {
    Uniform(BlockId),
    Paletted(Paletted),
}

impl Chunk {
    /// A chunk where every block is `id`.
    pub fn filled(id: BlockId) -> Self {
        Self {
            storage: Storage::Uniform(id),
        }
    }

    /// Builds a chunk from one id per block, in [`LocalPos`] index order.
    pub fn from_blocks(blocks: &[BlockId]) -> Self {
        assert_eq!(
            blocks.len(),
            CHUNK_VOLUME,
            "a chunk has {CHUNK_VOLUME} blocks"
        );
        let mut chunk = Self::filled(blocks[0]);
        for (index, &id) in blocks.iter().enumerate().skip(1) {
            chunk.set(LocalPos::from_index(index), id);
        }
        chunk
    }

    pub fn get(&self, pos: LocalPos) -> BlockId {
        match &self.storage {
            Storage::Uniform(id) => *id,
            Storage::Paletted(paletted) => paletted.get(pos.index()),
        }
    }

    /// Sets a block and returns the one it replaced.
    pub fn set(&mut self, pos: LocalPos, id: BlockId) -> BlockId {
        match &mut self.storage {
            Storage::Uniform(current) if *current == id => id,
            Storage::Uniform(current) => {
                let previous = *current;
                let mut paletted = Paletted::filled(previous);
                paletted.set(pos.index(), id);
                self.storage = Storage::Paletted(paletted);
                previous
            }
            Storage::Paletted(paletted) => paletted.set(pos.index(), id),
        }
    }

    /// The block filling the whole chunk, if it is made of a single block.
    pub fn uniform(&self) -> Option<BlockId> {
        match &self.storage {
            Storage::Uniform(id) => Some(*id),
            Storage::Paletted(_) => None,
        }
    }

    /// Writes every block into `out`, in [`LocalPos`] index order.
    pub fn copy_to(&self, out: &mut [BlockId]) {
        assert_eq!(out.len(), CHUNK_VOLUME, "a chunk has {CHUNK_VOLUME} blocks");
        match &self.storage {
            Storage::Uniform(id) => out.fill(*id),
            Storage::Paletted(paletted) => paletted.copy_to(out),
        }
    }
}

/// Bits per palette index: powers of two, so an index never straddles two words.
const INDEX_BITS: [u32; 6] = [1, 2, 4, 8, 16, 32];

#[derive(Clone, Debug, PartialEq, Eq)]
struct Paletted {
    palette: Vec<BlockId>,
    bits: u32,
    words: Box<[u64]>,
}

impl Paletted {
    /// One-bit indices, all pointing at `id`.
    fn filled(id: BlockId) -> Self {
        Self {
            palette: vec![id],
            bits: INDEX_BITS[0],
            words: vec![0; words_for(INDEX_BITS[0])].into(),
        }
    }

    fn get(&self, index: usize) -> BlockId {
        self.palette[self.index_at(index)]
    }

    fn set(&mut self, index: usize, id: BlockId) -> BlockId {
        let previous = self.get(index);
        if previous != id {
            let palette_index = self.palette_index(id);
            self.set_index_at(index, palette_index);
        }
        previous
    }

    fn copy_to(&self, out: &mut [BlockId]) {
        let per_word = 64 / self.bits as usize;
        let mask = index_mask(self.bits);
        for (word, blocks) in self.words.iter().zip(out.chunks_mut(per_word)) {
            for (slot, block) in blocks.iter_mut().enumerate() {
                let palette_index = (word >> (slot as u32 * self.bits)) & mask;
                *block = self.palette[palette_index as usize];
            }
        }
    }

    /// Position of `id` in the palette, adding it (and widening the indices
    /// when they run out of bits) if it is not there yet.
    fn palette_index(&mut self, id: BlockId) -> usize {
        if let Some(position) = self.palette.iter().position(|&known| known == id) {
            return position;
        }
        self.palette.push(id);
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

fn words_for(bits: u32) -> usize {
    CHUNK_VOLUME * bits as usize / 64
}

fn index_mask(bits: u32) -> u64 {
    (1u64 << bits) - 1
}

#[cfg(test)]
mod tests {
    use super::*;

    fn id(raw: u32) -> BlockId {
        BlockId::from_raw(raw)
    }

    #[test]
    fn stays_uniform_until_a_different_block_is_set() {
        let mut chunk = Chunk::filled(id(5));
        assert_eq!(chunk.set(LocalPos::new(1, 2, 3), id(5)), id(5));
        assert_eq!(chunk.uniform(), Some(id(5)));

        assert_eq!(chunk.set(LocalPos::new(1, 2, 3), id(7)), id(5));
        assert_eq!(chunk.uniform(), None);
        assert_eq!(chunk.get(LocalPos::new(1, 2, 3)), id(7));
        assert_eq!(chunk.get(LocalPos::new(0, 0, 0)), id(5));
    }

    #[test]
    fn widens_indices_as_the_palette_grows() {
        let mut chunk = Chunk::filled(BlockId::AIR);
        // 300 distinct blocks need 16-bit indices.
        for (n, pos) in LocalPos::all().step_by(97).take(300).enumerate() {
            chunk.set(pos, id(n as u32 + 10));
        }
        for (n, pos) in LocalPos::all().step_by(97).take(300).enumerate() {
            assert_eq!(chunk.get(pos), id(n as u32 + 10));
        }
        assert_eq!(chunk.get(LocalPos::new(31, 31, 31)), BlockId::AIR);
    }

    #[test]
    fn from_blocks_and_copy_to_round_trip() {
        let blocks: Vec<BlockId> = (0..CHUNK_VOLUME)
            .map(|i| id((i * 7919 % 13) as u32))
            .collect();
        let chunk = Chunk::from_blocks(&blocks);
        let mut copy = vec![BlockId::AIR; CHUNK_VOLUME];
        chunk.copy_to(&mut copy);
        assert_eq!(copy, blocks);
    }
}
