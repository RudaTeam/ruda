//! The client's side of the game: the world as the server has sent it, and
//! the player's actions on it.
//!
//! Actions and movement apply locally right away so the game feels instant;
//! the server either confirms them or sends back what the world really looks
//! like and where the player really is.

use std::collections::{HashMap, VecDeque};
use std::sync::Arc;
use std::time::Instant;

use glam::{DVec3, IVec3};
use ruda_core::{
    BlockId, BlockPos, CHUNK_SHIFT, CHUNK_SIZE, ChunkPos, Content, LocalPos, WorldBounds,
};
use ruda_net::{ClientConnection, Disconnected, RecvError};
use ruda_protocol::{
    ClientMessage, PROTOCOL_VERSION, PlayerInput, PlayerState, ServerMessage, TICK_RATE,
};
use ruda_sim::solid_in;

use ruda_world::light::LightEngine;
use ruda_world::lod::{LodTile, LodTilePos};
use ruda_world::{ChunkLight, World};

/// Something that happened since the last [`Client::update`].
#[derive(Clone, Debug, PartialEq)]
pub enum Event {
    /// The server let us in; where the player is follows shortly.
    Joined,
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

/// Where the player is, as the client works it out from its own inputs ahead
/// of the server's word.
#[derive(Clone, Debug)]
struct Prediction {
    /// After the last input.
    state: PlayerState,
    /// A tick earlier: the camera moves between the two.
    previous: PlayerState,
    /// Inputs sent that the server hasn't handled yet, oldest first.
    pending: VecDeque<PlayerInput>,
    /// How far the camera lags behind where the server's word moved the
    /// player, when it did: this fades out instead of the camera jumping.
    correction: DVec3,
    corrected_at: Instant,
}

impl Prediction {
    /// How long a correction takes to fade to a third.
    const FADE: f64 = 0.1;
    /// Larger corrections, like being moved by the server, are jumped to.
    const MOST_EASED: f64 = 4.0;

    fn new(state: PlayerState, now: Instant) -> Self {
        Self {
            state,
            previous: state,
            pending: VecDeque::new(),
            correction: DVec3::ZERO,
            corrected_at: now,
        }
    }

    fn correction(&self, now: Instant) -> DVec3 {
        let elapsed = now
            .saturating_duration_since(self.corrected_at)
            .as_secs_f64();
        self.correction * (-elapsed / Self::FADE).exp()
    }

    /// Where the feet are to be drawn, `alpha` of the way through the tick.
    fn position(&self, alpha: f64, now: Instant) -> DVec3 {
        self.previous.position.lerp(self.state.position, alpha) + self.correction(now)
    }

    /// Moves the player by an input the server has yet to handle.
    fn advance(&mut self, input: PlayerInput, solid: impl Fn(BlockPos) -> Option<bool>) {
        self.previous = self.state;
        self.state.step(&input, solid);
        self.pending.push_back(input);
    }

    /// Takes the server's word on where the player was after input `ack`,
    /// and works out from it where the inputs since then take the player.
    fn correct(
        &mut self,
        ack: u32,
        state: PlayerState,
        solid: impl Fn(BlockPos) -> Option<bool>,
        now: Instant,
    ) {
        while self.pending.front().is_some_and(|input| input.seq <= ack) {
            self.pending.pop_front();
        }
        let mut predicted = state;
        for input in &self.pending {
            predicted.step(input, &solid);
        }
        let error = self.state.position - predicted.position;
        if error.length() > Self::MOST_EASED {
            *self = Self {
                pending: std::mem::take(&mut self.pending),
                ..Self::new(predicted, now)
            };
            return;
        }
        // The camera stays where it is for now and eases over.
        if error != DVec3::ZERO {
            self.correction = self.correction(now) + error;
            self.corrected_at = now;
            self.previous.position -= error;
        }
        self.state = predicted;
    }
}

#[derive(Debug)]
pub struct Client {
    connection: ClientConnection,
    content: Arc<Content>,
    world: World,
    /// Spreads light after blocks change, as the server does, so the world
    /// doesn't show stale light until the server's arrives. Set once
    /// joined.
    light: Option<LightEngine>,
    lod: HashMap<LodTilePos, LodTile>,
    /// Set once the server has said where the player is.
    player: Option<Prediction>,
    last_input: u32,
    bounds: Option<WorldBounds>,
    time: Option<Clock>,
    sky_seed: u64,
    events: Vec<Event>,
    next_seq: u32,
    /// Actions the server has not answered yet.
    pending: usize,
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
            light: None,
            lod: HashMap::new(),
            player: None,
            last_input: 0,
            bounds: None,
            time: None,
            sky_seed: 0,
            events: Vec::new(),
            next_seq: 0,
            pending: 0,
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
                self.bounds = Some(bounds);
                self.light = Some(LightEngine::new(self.content.blocks(), bounds));
                self.events.push(Event::Joined);
            }
            ServerMessage::Disconnect { reason } => self.disconnect(&reason),
            ServerMessage::Player { ack, state } => {
                let now = Instant::now();
                let solid = solid_in(
                    &self.world,
                    self.content.blocks(),
                    self.bounds.unwrap_or_default(),
                );
                match &mut self.player {
                    Some(player) => player.correct(ack, state, solid, now),
                    None => self.player = Some(Prediction::new(state, now)),
                }
            }
            ServerMessage::Chunk { pos, chunk } => {
                self.world.insert_chunk(pos, chunk);
                if let Some(light) = &mut self.light {
                    light.mark_lit(pos);
                }
                self.events.push(Event::ChunkLoaded(pos));
            }
            ServerMessage::UnloadChunk(pos) => {
                if let Some(light) = &mut self.light {
                    light.forget(pos);
                }
                if self.world.remove_chunk(pos).is_some() {
                    self.events.push(Event::ChunkUnloaded(pos));
                }
            }
            ServerMessage::BlockChanged { pos, block } => {
                if let Some(previous) = self.world.set_block(pos, block)
                    && previous != block
                {
                    self.events.push(Event::BlockChanged(pos));
                    self.relight(pos, previous);
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

    /// Moves the player by this tick's input right away, and sends the input
    /// to the server; its `seq` is filled in. Call once a tick. Does nothing
    /// until the server has said where the player is.
    pub fn tick_player(&mut self, mut input: PlayerInput) {
        let solid = solid_in(
            &self.world,
            self.content.blocks(),
            self.bounds.unwrap_or_default(),
        );
        let Some(player) = &mut self.player else {
            return;
        };
        self.last_input += 1;
        input.seq = self.last_input;
        player.advance(input, solid);
        self.send(&ClientMessage::Input(input));
    }

    /// Where the player is after its last input, once the server has said
    /// where it starts.
    pub fn player(&self) -> Option<&PlayerState> {
        Some(&self.player.as_ref()?.state)
    }

    /// Where the player's feet are to be drawn, `alpha` of the way from the
    /// tick before the last to the last.
    pub fn player_position(&self, alpha: f64) -> Option<DVec3> {
        Some(self.player.as_ref()?.position(alpha, Instant::now()))
    }

    /// Whether a jump would take the player onto the block it is walking
    /// into with `input`; see [`PlayerState::should_auto_jump`].
    pub fn should_auto_jump(&self, input: &PlayerInput) -> bool {
        let solid = solid_in(
            &self.world,
            self.content.blocks(),
            self.bounds.unwrap_or_default(),
        );
        self.player()
            .is_some_and(|player| player.should_auto_jump(input, solid))
    }

    /// Inputs sent that the server has not handled yet.
    pub fn pending_inputs(&self) -> usize {
        self.player
            .as_ref()
            .map_or(0, |player| player.pending.len())
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
    /// hold on to, and a solid block can't go where the player is. Returns
    /// false if that isn't possible.
    pub fn place_block(&mut self, pos: BlockPos, block: BlockId) -> bool {
        let blocks = self.content.blocks();
        let empty = self.world.block(pos) == Some(BlockId::AIR);
        let inside = self.bounds.is_some_and(|bounds| bounds.contains(pos));
        let placeable =
            block != BlockId::UNKNOWN && blocks.is_visible(block) && blocks.is_breakable(block);
        let supported = blocks
            .support(block)
            .is_none_or(|face| self.is_solid(pos.offset(face)));
        let in_player =
            blocks.is_solid(block) && self.player().is_some_and(|player| player.overlaps(pos));
        if !empty || !inside || !placeable || !supported || in_player {
            return false;
        }
        let seq = self.next_seq();
        self.act(pos, block, ClientMessage::PlaceBlock { pos, block, seq });
        true
    }

    fn act(&mut self, pos: BlockPos, block: BlockId, message: ClientMessage) {
        if let Some(previous) = self.world.set_block(pos, block) {
            self.events.push(Event::BlockChanged(pos));
            self.relight(pos, previous);
        }
        self.pending += 1;
        self.send(&message);
    }

    /// Spreads light after the block at `pos`, once `old`, changed, as the
    /// server will. Its word on the light follows and wins.
    fn relight(&mut self, pos: BlockPos, old: BlockId) {
        let (Some(light), Some(bounds)) = (&mut self.light, self.bounds) else {
            return;
        };
        // The light of the columns it can change, to tell which chunks look
        // different afterwards.
        let center = pos.chunk().0;
        let heights = (bounds.min_y >> CHUNK_SHIFT)..=(bounds.max_y >> CHUNK_SHIFT);
        let before: Vec<(ChunkPos, ChunkLight)> = heights
            .flat_map(|y| (-1..=1).flat_map(move |z| (-1..=1).map(move |x| (x, y, z))))
            .map(|(x, y, z)| ChunkPos::new(center.x + x, y, center.z + z))
            .filter_map(|pos| Some((pos, self.world.chunk(pos)?.light().clone())))
            .collect();
        let changed = light.block_changed(&mut self.world, pos, old);
        let mut affected: Vec<ChunkPos> = Vec::new();
        for (pos, old) in before.iter().filter(|(pos, _)| changed.contains(pos)) {
            if let Some(chunk) = self.world.chunk(*pos) {
                affected.extend(affected_by(*pos, old, chunk.light()));
            }
        }
        affected.sort_by_key(|pos| (pos.0.x, pos.0.y, pos.0.z));
        affected.dedup();
        if !affected.is_empty() {
            self.events.push(Event::LightChanged(affected));
        }
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
    use std::time::Duration;

    use glam::Vec2;
    use ruda_core::Light;

    use super::*;

    /// Ground below y = 0.
    fn flat(pos: BlockPos) -> Option<bool> {
        Some(pos.0.y < 0)
    }

    fn walk(seq: u32) -> PlayerInput {
        PlayerInput {
            seq,
            walk: Vec2::Y,
            ..Default::default()
        }
    }

    /// A player on the ground that has walked four ticks, and where the
    /// server saw it after the first two.
    fn walked() -> (Prediction, PlayerState, Instant) {
        let now = Instant::now();
        let mut start = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        start.step(&PlayerInput::default(), flat);
        start.step(&PlayerInput::default(), flat);
        let mut prediction = Prediction::new(start, now);
        let mut server = start;
        for seq in 1..=4 {
            prediction.advance(walk(seq), flat);
            if seq <= 2 {
                server.step(&walk(seq), flat);
            }
        }
        (prediction, server, now)
    }

    #[test]
    fn keeps_its_prediction_when_the_server_agrees() {
        let (mut prediction, server, now) = walked();
        let before = prediction.clone();
        prediction.correct(2, server, flat, now);
        assert_eq!(prediction.state, before.state);
        assert_eq!(prediction.pending.len(), 2);
        assert_eq!(prediction.correction(now), DVec3::ZERO);
    }

    #[test]
    fn eases_over_to_where_the_server_puts_the_player() {
        let (mut prediction, mut server, now) = walked();
        let drawn = prediction.position(0.5, now);
        // Something pushed the player a block aside.
        server.position.x += 1.0;
        prediction.correct(2, server, flat, now);
        assert!((prediction.state.position.x - 1.5).abs() < 1e-9);
        assert!((prediction.position(0.5, now) - drawn).length() < 1e-9);
        let later = now + Duration::from_secs(1);
        assert!((prediction.position(0.5, later).x - 1.5).abs() < 1e-3);

        // Far moves are jumped to.
        server.position.x += 10.0;
        prediction.correct(3, server, flat, now);
        assert_eq!(prediction.pending.len(), 1);
        assert!((prediction.position(0.0, now).x - 11.5).abs() < 1e-9);
    }

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
