//! The client's side of the game: the world as the server has sent it, and
//! the player's actions on it.
//!
//! Actions apply locally right away so the game feels instant; the server
//! either confirms them or sends back what the world really looks like.

use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use ruda_core::{BlockId, BlockPos, CHUNK_SIZE, ChunkPos, Content, LocalPos, WorldBounds};
use ruda_net::{ClientConnection, Disconnected, RecvError};
use ruda_protocol::{ClientMessage, PROTOCOL_VERSION, ServerMessage, TICK_RATE};
use std::collections::HashMap;

use ruda_world::lod::{LodTile, LodTilePos};
use ruda_world::{ChunkLight, World};

/// Something that happened since the last [`Client::update`].
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The server let us in; the player starts at `spawn`.
    Joined {
        spawn: DVec3,
    },
    ChunkLoaded(ChunkPos),
    ChunkUnloaded(ChunkPos),
    /// A block changed, by this player's action or anybody else's.
    BlockChanged(BlockPos),
    /// Light changed; these chunks look different now, as their own light
    /// or that of blocks right next to them changed.
    LightChanged(Vec<ChunkPos>),
    /// The far-away look of a tile arrived, or changed.
    LodLoaded(LodTilePos),
    LodUnloaded(LodTilePos),
    /// The connection is over; no events follow.
    Disconnected {
        reason: String,
    },
}

/// The world's time: the server's last word on it, counted on by the
/// client's own clock.
///
/// The two clocks drift a little apart, so the server's time and the
/// client's count differ by a fraction of a tick at each update. Instead of
/// jumping to the server's time, which shakes everything that moves with it
/// (the sun, its shadows, the clouds), the count eases over to it.
#[derive(Clone, Copy, Debug)]
struct Clock {
    ticks: u64,
    at: Instant,
    /// How far ahead of `ticks` the count was when they arrived; this fades
    /// out over [`Clock::EASE`].
    ahead: f64,
}

impl Clock {
    const EASE: f64 = 1.0;
    /// Larger differences, like the time being set, are jumped to.
    const MOST_EASED: f64 = 0.5 * TICK_RATE as f64;

    fn new(previous: Option<Self>, ticks: u64, now: Instant) -> Self {
        let ahead = previous
            .map(|clock| clock.at(now) - ticks as f64)
            .filter(|ahead| ahead.abs() < Self::MOST_EASED)
            .unwrap_or(0.0);
        Self {
            ticks,
            at: now,
            ahead,
        }
    }

    fn at(&self, now: Instant) -> f64 {
        let elapsed = now.saturating_duration_since(self.at).as_secs_f64();
        let easing = (1.0 - elapsed / Self::EASE).max(0.0);
        self.ticks as f64 + elapsed * f64::from(TICK_RATE) + self.ahead * easing
    }
}

#[derive(Debug)]
pub struct Client {
    connection: ClientConnection,
    content: Arc<Content>,
    world: World,
    lod: HashMap<LodTilePos, LodTile>,
    spawn: Option<DVec3>,
    bounds: Option<WorldBounds>,
    time: Option<Clock>,
    sky_seed: u64,
    events: Vec<Event>,
    next_seq: u32,
    /// Actions the server has not answered yet.
    pending: usize,
    last_position: Option<DVec3>,
    connected: bool,
}

impl Client {
    /// Introduces itself to the server; the reply arrives as [`Event::Joined`].
    pub fn connect(
        connection: ClientConnection,
        content: Arc<Content>,
        name: &str,
    ) -> Result<Self, Disconnected> {
        connection.send(&ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            name: name.to_owned(),
        })?;
        Ok(Self {
            connection,
            content,
            world: World::new(),
            lod: HashMap::new(),
            spawn: None,
            bounds: None,
            time: None,
            sky_seed: 0,
            events: Vec::new(),
            next_seq: 0,
            pending: 0,
            last_position: None,
            connected: true,
        })
    }

    /// Applies everything the server has sent and returns what changed.
    pub fn update(&mut self) -> std::vec::Drain<'_, Event> {
        while self.connected {
            match self.connection.try_recv() {
                Ok(Some(message)) => self.handle(message),
                Ok(None) => break,
                Err(RecvError::Disconnected) => {
                    self.disconnect("lost the connection to the server")
                }
                Err(RecvError::Malformed(error)) => self.disconnect(&error.to_string()),
            }
        }
        self.events.drain(..)
    }

    fn handle(&mut self, message: ServerMessage) {
        match message {
            ServerMessage::Welcome {
                blocks,
                spawn,
                bounds,
                time,
                sky_seed,
            } => {
                self.time = Some(Clock::new(None, time, Instant::now()));
                self.sky_seed = sky_seed;
                let ours = self.content.blocks().iter().map(|(_, def)| &def.id);
                if !ours.eq(blocks.iter()) {
                    return self.disconnect("this game's content differs from the server's");
                }
                self.spawn = Some(spawn);
                self.bounds = Some(bounds);
                self.events.push(Event::Joined { spawn });
            }
            ServerMessage::Disconnect { reason } => self.disconnect(&reason),
            ServerMessage::Chunk { pos, chunk } => {
                self.world.insert_chunk(pos, chunk);
                self.events.push(Event::ChunkLoaded(pos));
            }
            ServerMessage::UnloadChunk(pos) => {
                if self.world.remove_chunk(pos).is_some() {
                    self.events.push(Event::ChunkUnloaded(pos));
                }
            }
            ServerMessage::BlockChanged { pos, block } => {
                if self
                    .world
                    .set_block(pos, block)
                    .is_some_and(|previous| previous != block)
                {
                    self.events.push(Event::BlockChanged(pos));
                }
            }
            ServerMessage::Light { pos, light } => {
                if let Some(chunk) = self.world.chunk_mut(pos) {
                    let affected = affected_by(pos, chunk.light(), &light);
                    chunk.set_light(light);
                    if !affected.is_empty() {
                        self.events.push(Event::LightChanged(affected));
                    }
                }
            }
            ServerMessage::Time(time) => self.set_time(time, Instant::now()),
            ServerMessage::LodTile { pos, tile } => {
                if !tile.is_complete() {
                    return self.disconnect("the server sent a malformed tile");
                }
                self.lod.insert(pos, tile);
                self.events.push(Event::LodLoaded(pos));
            }
            ServerMessage::UnloadLod(pos) => {
                if self.lod.remove(&pos).is_some() {
                    self.events.push(Event::LodUnloaded(pos));
                }
            }
            ServerMessage::ActionDone { .. } => self.pending = self.pending.saturating_sub(1),
        }
    }

    /// Tells the server where the player is, if it moved noticeably.
    pub fn set_position(&mut self, position: DVec3) {
        let moved = self
            .last_position
            .is_none_or(|last| last.distance_squared(position) > 0.01);
        if moved {
            self.last_position = Some(position);
            self.send(&ClientMessage::Position(position));
        }
    }

    /// Asks the server to stream the world this far around the player, in
    /// chunks.
    pub fn set_view_distance(&mut self, chunks: u8) {
        self.send(&ClientMessage::ViewDistance(chunks));
    }

    /// Asks the server for the far-away look of the world this far around
    /// the player, in blocks; 0 for none.
    pub fn set_lod_distance(&mut self, blocks: u16) {
        self.send(&ClientMessage::LodDistance(blocks));
    }

    /// The far-away look of a tile, if the server sent it.
    pub fn lod(&self, pos: LodTilePos) -> Option<&LodTile> {
        self.lod.get(&pos)
    }

    /// Breaks a block. Returns false if there is nothing that can be broken.
    pub fn break_block(&mut self, pos: BlockPos) -> bool {
        let blocks = self.content.blocks();
        let breakable = self
            .world
            .block(pos)
            .is_some_and(|block| blocks.is_visible(block) && blocks.is_breakable(block));
        if !breakable {
            return false;
        }
        let seq = self.next_seq();
        self.act(pos, BlockId::AIR, ClientMessage::BreakBlock { pos, seq });
        true
    }

    /// Places `block` into an empty cell; a torch needs a solid block to
    /// hold on to. Returns false if that isn't possible.
    pub fn place_block(&mut self, pos: BlockPos, block: BlockId) -> bool {
        let blocks = self.content.blocks();
        let empty = self.world.block(pos) == Some(BlockId::AIR);
        let inside = self.bounds.is_some_and(|bounds| bounds.contains(pos));
        let placeable =
            block != BlockId::UNKNOWN && blocks.is_visible(block) && blocks.is_breakable(block);
        let supported = blocks
            .support(block)
            .is_none_or(|face| self.is_solid(pos.offset(face)));
        if !empty || !inside || !placeable || !supported {
            return false;
        }
        let seq = self.next_seq();
        self.act(pos, block, ClientMessage::PlaceBlock { pos, block, seq });
        true
    }

    fn act(&mut self, pos: BlockPos, block: BlockId, message: ClientMessage) {
        self.world.set_block(pos, block);
        self.events.push(Event::BlockChanged(pos));
        self.pending += 1;
        self.send(&message);
    }

    fn next_seq(&mut self) -> u32 {
        self.next_seq = self.next_seq.wrapping_add(1);
        self.next_seq
    }

    fn send(&mut self, message: &ClientMessage) {
        if self.connected && self.connection.send(message).is_err() {
            self.disconnect("lost the connection to the server");
        }
    }

    fn disconnect(&mut self, reason: &str) {
        if self.connected {
            self.connected = false;
            self.events.push(Event::Disconnected {
                reason: reason.to_owned(),
            });
        }
    }

    /// Whether the block at `pos` is loaded and can be aimed at: anything
    /// but air.
    pub fn is_targetable(&self, pos: BlockPos) -> bool {
        self.world
            .block(pos)
            .is_some_and(|block| self.content.blocks().is_visible(block))
    }

    /// Whether the block at `pos` is loaded and solid.
    pub fn is_solid(&self, pos: BlockPos) -> bool {
        self.world
            .block(pos)
            .is_some_and(|block| self.content.blocks().is_solid(block))
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn content(&self) -> &Arc<Content> {
        &self.content
    }

    /// Where the server put the player, once joined.
    pub fn spawn(&self) -> Option<DVec3> {
        self.spawn
    }

    /// The world's time in ticks, counting on smoothly between the server's
    /// updates; see [`ruda_protocol::DAY_LENGTH`].
    pub fn time(&self) -> Option<f64> {
        self.time.map(|clock| clock.at(Instant::now()))
    }

    fn set_time(&mut self, ticks: u64, now: Instant) {
        self.time = Some(Clock::new(self.time, ticks, now));
    }

    /// Decides the clouds, once joined.
    pub fn sky_seed(&self) -> u64 {
        self.sky_seed
    }

    /// The heights blocks can exist at, once joined.
    pub fn bounds(&self) -> Option<WorldBounds> {
        self.bounds
    }

    /// Actions sent that the server has not answered yet.
    pub fn pending_actions(&self) -> usize {
        self.pending
    }

    pub fn is_connected(&self) -> bool {
        self.connected
    }
}

/// Chunks whose looks depend on light that changed from `old` to `new` in
/// the chunk at `pos`: the chunk itself, and neighbours touching blocks
/// whose light changed.
fn affected_by(pos: ChunkPos, old: &ChunkLight, new: &ChunkLight) -> Vec<ChunkPos> {
    if old == new {
        return Vec::new();
    }
    let last = CHUNK_SIZE as u32 - 1;
    let mut near = [false; 27];
    for local in LocalPos::all() {
        if old.get(local) == new.get(local) {
            continue;
        }
        let range = |c: u32| (if c == 0 { -1 } else { 0 })..=(if c == last { 1 } else { 0 });
        for y in range(local.y()) {
            for z in range(local.z()) {
                for x in range(local.x()) {
                    near[((y + 1) * 9 + (z + 1) * 3 + (x + 1)) as usize] = true;
                }
            }
        }
    }
    (0..27)
        .filter(|&index| near[index])
        .map(|index| {
            let index = index as i32;
            let offset = IVec3::new(index % 3 - 1, index / 9 - 1, index / 3 % 3 - 1);
            ChunkPos(pos.0 + offset)
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use ruda_core::Light;

    use super::*;

    #[test]
    fn light_changes_reach_the_neighbours_they_touch() {
        let pos = ChunkPos::new(2, 0, -1);
        let old = ChunkLight::uniform(Light::DARK);

        let mut middle = old.clone();
        middle.set(LocalPos::new(10, 10, 10), Light::SKY);
        assert_eq!(affected_by(pos, &old, &middle), vec![pos]);

        let mut corner = old.clone();
        corner.set(LocalPos::new(0, 31, 5), Light::SKY);
        let mut affected = affected_by(pos, &old, &corner);
        affected.sort_by_key(|p| (p.0.x, p.0.y, p.0.z));
        assert_eq!(
            affected,
            vec![
                ChunkPos::new(1, 0, -1),
                ChunkPos::new(1, 1, -1),
                pos,
                ChunkPos::new(2, 1, -1),
            ]
        );
        assert!(affected_by(pos, &old, &old).is_empty());
    }

    #[test]
    fn the_clock_eases_over_to_the_servers_time() {
        let start = Instant::now();
        let later = |seconds: f64| start + std::time::Duration::from_secs_f64(seconds);
        let rate = f64::from(TICK_RATE);
        let clock = Clock::new(None, 1000, start);
        assert_eq!(clock.at(later(1.0)), 1000.0 + rate);

        // A second later the server says it's a tick behind the count: the
        // count doesn't jump back, and in a second it agrees with the server.
        let behind = 1000 + TICK_RATE as u64 - 1;
        let eased = Clock::new(Some(clock), behind, later(1.0));
        assert_eq!(eased.at(later(1.0)), clock.at(later(1.0)));
        assert!(eased.at(later(1.5)) > eased.at(later(1.4)));
        assert_eq!(eased.at(later(2.0)), behind as f64 + rate);

        // Setting the time jumps.
        let set = Clock::new(Some(clock), 9000, later(1.0));
        assert_eq!(set.at(later(1.0)), 9000.0);
    }
}
