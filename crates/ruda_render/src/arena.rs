//! Room in large shared buffers, so chunks don't each need a buffer of their
//! own and neighbouring draws can be merged.

use std::ops::Range;

/// Hands out ranges of a fixed-size space, first fit.
#[derive(Clone, Debug)]
pub(crate) struct RangeAllocator {
    /// Free ranges, sorted and never touching each other.
    free: Vec<Range<u32>>,
}

impl RangeAllocator {
    pub(crate) fn new(size: u32) -> Self {
        let mut free = Vec::new();
        if size > 0 {
            free.push(0..size);
        }
        Self { free }
    }

    pub(crate) fn allocate(&mut self, len: u32) -> Option<Range<u32>> {
        let index = self
            .free
            .iter()
            .position(|range| range.end - range.start >= len)?;
        let range = &mut self.free[index];
        let start = range.start;
        range.start += len;
        if range.start == range.end {
            self.free.remove(index);
        }
        Some(start..start + len)
    }

    pub(crate) fn free(&mut self, range: Range<u32>) {
        if range.start == range.end {
            return;
        }
        let index = self.free.partition_point(|free| free.start < range.start);
        let joins_previous = index > 0 && self.free[index - 1].end == range.start;
        let joins_next = index < self.free.len() && self.free[index].start == range.end;
        match (joins_previous, joins_next) {
            (true, true) => {
                self.free[index - 1].end = self.free[index].end;
                self.free.remove(index);
            }
            (true, false) => self.free[index - 1].end = range.end,
            (false, true) => self.free[index].start = range.start,
            (false, false) => self.free.insert(index, range),
        }
    }

    /// Whether nothing is allocated.
    #[cfg(test)]
    fn is_unused(&self, size: u32) -> bool {
        self.free.len() == 1 && self.free[0] == (0..size)
    }
}

/// Small integers handed out and taken back, lowest first.
#[derive(Clone, Debug, Default)]
pub(crate) struct Slots {
    next: u32,
    free: Vec<u32>,
    limit: u32,
}

impl Slots {
    pub(crate) fn new(limit: u32) -> Self {
        Self {
            next: 0,
            free: Vec::new(),
            limit,
        }
    }

    pub(crate) fn take(&mut self) -> Option<u32> {
        if let Some(slot) = self.free.pop() {
            return Some(slot);
        }
        (self.next < self.limit).then(|| {
            self.next += 1;
            self.next - 1
        })
    }

    pub(crate) fn give_back(&mut self, slot: u32) {
        self.free.push(slot);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn allocates_and_merges_freed_ranges() {
        let mut ranges = RangeAllocator::new(100);
        let a = ranges.allocate(10).unwrap();
        let b = ranges.allocate(20).unwrap();
        let c = ranges.allocate(30).unwrap();
        assert_eq!((a.clone(), b.clone(), c.clone()), (0..10, 10..30, 30..60));
        assert_eq!(ranges.allocate(50), None);

        ranges.free(b);
        // First fit: the hole left by b.
        assert_eq!(ranges.allocate(5), Some(10..15));
        ranges.free(10..15);
        ranges.free(a);
        ranges.free(c);
        assert!(ranges.is_unused(100));
        assert_eq!(ranges.allocate(100), Some(0..100));
    }

    #[test]
    fn reuses_slots() {
        let mut slots = Slots::new(2);
        assert_eq!(
            (slots.take(), slots.take(), slots.take()),
            (Some(0), Some(1), None)
        );
        slots.give_back(0);
        assert_eq!(slots.take(), Some(0));
    }
}
