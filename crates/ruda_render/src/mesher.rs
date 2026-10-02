use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};

use ruda_core::ChunkPos;
use ruda_world::World;

use crate::{BlockFaces, ChunkMesh, PaddedChunk, mesh_chunk};

/// Keeps chunk meshes up to date: chunks marked dirty are meshed on the
/// worker pool, nearest to the camera first.
#[derive(Debug)]
pub struct ChunkMesher {
    faces: Arc<BlockFaces>,
    dirty: HashSet<ChunkPos>,
    /// Chunks being meshed, with the version of the job.
    in_flight: HashMap<ChunkPos, u64>,
    next_version: u64,
    max_in_flight: usize,
    done_tx: Sender<(ChunkPos, u64, ChunkMesh)>,
    done_rx: Receiver<(ChunkPos, u64, ChunkMesh)>,
}

impl ChunkMesher {
    pub fn new(faces: Arc<BlockFaces>) -> Self {
        let (done_tx, done_rx) = mpsc::channel();
        Self {
            faces,
            dirty: HashSet::new(),
            in_flight: HashMap::new(),
            next_version: 0,
            max_in_flight: rayon::current_num_threads() * 2,
            done_tx,
            done_rx,
        }
    }

    /// The chunk's geometry is out of date.
    pub fn mark_dirty(&mut self, pos: ChunkPos) {
        self.dirty.insert(pos);
    }

    /// A newly loaded chunk: its neighbours' faces towards it may have
    /// appeared or disappeared too, and the shading of their corners changed.
    pub fn chunk_loaded(&mut self, pos: ChunkPos) {
        for y in -1..=1 {
            for z in -1..=1 {
                for x in -1..=1 {
                    self.mark_dirty(ChunkPos(pos.0 + glam::IVec3::new(x, y, z)));
                }
            }
        }
    }

    /// The chunk is gone; any result for it is no longer wanted.
    pub fn forget(&mut self, pos: ChunkPos) {
        self.dirty.remove(&pos);
        self.in_flight.remove(&pos);
    }

    /// Starts meshing dirty chunks nearest to `center`, as many as the
    /// worker budget allows. Chunks missing from the world are dropped.
    pub fn schedule(&mut self, world: &World, center: ChunkPos) {
        let budget = self.max_in_flight.saturating_sub(self.in_flight.len());
        if budget == 0 || self.dirty.is_empty() {
            return;
        }
        let mut ready: Vec<ChunkPos> = self
            .dirty
            .iter()
            .filter(|pos| !self.in_flight.contains_key(pos))
            .copied()
            .collect();
        ready.sort_by_key(|pos| (pos.0 - center.0).length_squared());
        for pos in ready.into_iter().take(budget) {
            self.dirty.remove(&pos);
            let Some(chunk) = PaddedChunk::gather(world, pos) else {
                continue;
            };
            self.next_version += 1;
            let version = self.next_version;
            self.in_flight.insert(pos, version);
            let faces = Arc::clone(&self.faces);
            let done = self.done_tx.clone();
            rayon::spawn(move || {
                let _ = done.send((pos, version, mesh_chunk(&chunk, &faces)));
            });
        }
    }

    /// Meshes finished since the last call, minus those that went stale.
    pub fn finished(&mut self) -> Vec<(ChunkPos, ChunkMesh)> {
        let mut meshes = Vec::new();
        while let Ok((pos, version, mesh)) = self.done_rx.try_recv() {
            if self.in_flight.get(&pos) == Some(&version) {
                self.in_flight.remove(&pos);
                meshes.push((pos, mesh));
            }
        }
        meshes
    }

    /// Whether the chunk waits for or is being meshed.
    pub fn is_pending(&self, pos: ChunkPos) -> bool {
        self.dirty.contains(&pos) || self.in_flight.contains_key(&pos)
    }

    /// Chunks waiting for or being meshed.
    pub fn backlog(&self) -> usize {
        self.dirty.len() + self.in_flight.len()
    }
}
