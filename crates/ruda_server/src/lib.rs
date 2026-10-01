//! The authoritative game server.
//!
//! The server owns the world: clients only ask for changes and the server
//! decides. Single-player runs it on a thread next to the client, the
//! dedicated server on its own.

use std::collections::{HashMap, HashSet};
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glam::{DVec3, IVec3};
use ruda_core::{BlockId, BlockPos, ChunkPos, Content, Face, WorldBounds};
use ruda_net::{ClientConnection, RecvError, ServerConnection, local_pair};
pub use ruda_protocol::TICK_RATE;
use ruda_protocol::{ClientMessage, DAY_LENGTH, PROTOCOL_VERSION, REACH, ServerMessage};
use ruda_world::light::LightEngine;
use ruda_world::lod::{LOD_TILE_SIZE, LodTile, LodTilePos};
use ruda_world::{Chunk, Generator, World};
use tracing::{info, warn};

const TICK: Duration = Duration::from_millis(1000 / TICK_RATE as u64);

/// Most chunks sent to one client in a tick.
const CHUNKS_PER_TICK: usize = 64;
/// Most far-away tiles sent to one client in a tick.
const LOD_TILES_PER_TICK: usize = 4;
/// Time a tick may spend lighting new chunks.
const LIGHT_BUDGET: Duration = Duration::from_millis(10);

#[derive(Clone, Copy, Debug)]
pub struct ServerConfig {
    /// The largest radius, in chunks, of the area a player can ask to have
    /// streamed around it.
    pub view_distance: i32,
    /// Blocks exist only between these heights.
    pub bounds: WorldBounds,
    /// The time the world starts at, in ticks; see [`DAY_LENGTH`].
    pub start_time: u64,
    /// The farthest, in blocks, that the far-away look of the world is sent.
    pub max_lod_distance: i32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            view_distance: 16,
            bounds: WorldBounds::DEFAULT,
            // Early morning.
            start_time: DAY_LENGTH / 24,
            max_lod_distance: 2048,
        }
    }
}

/// View distance of a player who hasn't asked for one.
const DEFAULT_VIEW_DISTANCE: i32 = 6;

/// Worlds are far wider than they are tall, so the streamed area is half as
/// high as it is wide.
fn vertical_view_distance(radius: i32) -> i32 {
    (radius / 2).max(1)
}

pub struct Server {
    config: ServerConfig,
    content: Arc<Content>,
    generator: Arc<dyn Generator>,
    world: World,
    /// Chunks players have changed. Until there is saving they stay in
    /// memory for as long as the server runs.
    modified: HashSet<ChunkPos>,
    generating: HashSet<ChunkPos>,
    max_generating: usize,
    generated_tx: Sender<(ChunkPos, Chunk)>,
    generated_rx: Receiver<(ChunkPos, Chunk)>,
    light: LightEngine,
    /// Generated chunks waiting for their light.
    unlit: HashSet<ChunkPos>,
    /// Chunks whose light changed since clients last heard.
    light_changed: HashSet<ChunkPos>,
    /// The far-away look of tiles near players.
    lod_tiles: HashMap<LodTilePos, Arc<LodTile>>,
    lod_generating: HashSet<LodTilePos>,
    lod_tx: Sender<(LodTilePos, Option<LodTile>)>,
    lod_rx: Receiver<(LodTilePos, Option<LodTile>)>,
    /// Set when the generator can't tell its surface: no far-away look.
    lod_unsupported: bool,
    clients: Vec<RemoteClient>,
    /// Offsets of the chunks streamed around a player, nearest first, by
    /// view distance.
    views: HashMap<i32, Arc<[IVec3]>>,
    spawn: DVec3,
    ticks: u64,
    /// Ticks since the world began.
    time: u64,
}

struct RemoteClient {
    connection: ServerConnection,
    /// Set once the client has said hello.
    name: Option<String>,
    position: Option<DVec3>,
    /// In chunks.
    view_distance: i32,
    sent: HashSet<ChunkPos>,
    /// In blocks; 0 for no far-away look.
    lod_distance: i32,
    lod_sent: HashSet<LodTilePos>,
    connected: bool,
}

impl RemoteClient {
    fn send(&mut self, message: ServerMessage) {
        if self.connection.send(&message).is_err() {
            self.connected = false;
        }
    }
}

impl Server {
    pub fn new(content: Arc<Content>, generator: Arc<dyn Generator>, config: ServerConfig) -> Self {
        let spawn = match generator.surface_height(0, 0) {
            Some(height) => DVec3::new(0.5, f64::from(height) + 3.0, 0.5),
            None => DVec3::new(0.5, 100.0, 0.5),
        };
        let (generated_tx, generated_rx) = mpsc::channel();
        let (lod_tx, lod_rx) = mpsc::channel();
        Self {
            lod_tiles: HashMap::new(),
            lod_generating: HashSet::new(),
            lod_tx,
            lod_rx,
            lod_unsupported: false,
            views: HashMap::new(),
            light: LightEngine::new(content.blocks(), config.bounds),
            unlit: HashSet::new(),
            light_changed: HashSet::new(),
            time: config.start_time,
            config,
            content,
            generator,
            world: World::new(),
            modified: HashSet::new(),
            generating: HashSet::new(),
            // Enough to keep every core busy while new requests still go out
            // nearest first.
            max_generating: rayon::current_num_threads() * 4,
            generated_tx,
            generated_rx,
            clients: Vec::new(),
            spawn,
            ticks: 0,
        }
    }

    pub fn connect(&mut self, connection: ServerConnection) {
        self.clients.push(RemoteClient {
            connection,
            name: None,
            position: None,
            view_distance: DEFAULT_VIEW_DISTANCE.min(self.config.view_distance),
            sent: HashSet::new(),
            lod_distance: 0,
            lod_sent: HashSet::new(),
            connected: true,
        });
    }

    pub fn world(&self) -> &World {
        &self.world
    }

    pub fn spawn(&self) -> DVec3 {
        self.spawn
    }

    pub fn client_count(&self) -> usize {
        self.clients.len()
    }

    /// One simulation step.
    pub fn tick(&mut self) {
        while let Ok((pos, chunk)) = self.generated_rx.try_recv() {
            self.generating.remove(&pos);
            self.world.insert_chunk(pos, chunk);
            self.unlit.insert(pos);
        }
        while let Ok((pos, tile)) = self.lod_rx.try_recv() {
            self.lod_generating.remove(&pos);
            match tile {
                Some(tile) => drop(self.lod_tiles.insert(pos, Arc::new(tile))),
                None => self.lod_unsupported = true,
            }
        }
        for index in 0..self.clients.len() {
            self.handle_messages(index);
        }
        self.clients.retain(|client| client.connected);
        self.light_chunks();
        // Before streaming: chunks sent below already carry their new light.
        self.send_light_changes();
        for index in 0..self.clients.len() {
            self.stream_chunks(index);
            self.stream_lod(index);
        }
        self.ticks += 1;
        self.time += 1;
        if self.ticks.is_multiple_of(u64::from(TICK_RATE)) {
            let time = self.time;
            for client in self.clients.iter_mut().filter(|c| c.name.is_some()) {
                client.send(ServerMessage::Time(time));
            }
            self.unload_unused_chunks();
        }
    }

    /// The world's time, in ticks since it began.
    pub fn time(&self) -> u64 {
        self.time
    }

    /// Lights generated chunks whose chunk above is lit, top down, for as
    /// long as the tick's budget allows, and asks for the chunks above the
    /// others.
    fn light_chunks(&mut self) {
        let deadline = Instant::now() + LIGHT_BUDGET;
        loop {
            let mut ready: Vec<ChunkPos> = self
                .unlit
                .iter()
                .filter(|&&pos| self.light.can_light(&self.world, pos))
                .copied()
                .collect();
            if ready.is_empty() {
                break;
            }
            // Higher chunks first: they let the ones below be lit.
            ready.sort_by_key(|pos| std::cmp::Reverse(pos.0.y));
            for pos in ready {
                if Instant::now() > deadline {
                    return;
                }
                self.unlit.remove(&pos);
                let changed = self.light.light_chunk(&mut self.world, pos);
                self.light_changed.extend(changed);
            }
        }
        let missing: Vec<ChunkPos> = self
            .unlit
            .iter()
            .filter_map(|&pos| self.light.needs_above(pos))
            .filter(|above| self.world.chunk(*above).is_none())
            .collect();
        for above in missing {
            self.generate(above);
        }
    }

    /// Sends new light to the clients that have the chunks it changed in.
    fn send_light_changes(&mut self) {
        for pos in std::mem::take(&mut self.light_changed) {
            let Some(chunk) = self.world.chunk(pos) else {
                continue;
            };
            for client in &mut self.clients {
                if client.sent.contains(&pos) {
                    client.send(ServerMessage::Light {
                        pos,
                        light: chunk.light().clone(),
                    });
                }
            }
        }
    }

    /// Whether a chunk's light is final: it and every chunk around it that
    /// is inside the world are lit.
    fn is_complete(&self, pos: ChunkPos) -> bool {
        around(pos).all(|near| !self.config.bounds.contains_chunk(near) || self.light.is_lit(near))
    }

    /// Runs at [`TICK_RATE`] until every client has left.
    pub fn run_until_empty(mut self) {
        let mut deadline = Instant::now();
        while !self.clients.is_empty() {
            self.tick();
            deadline += TICK;
            let now = Instant::now();
            if let Some(wait) = deadline.checked_duration_since(now) {
                thread::sleep(wait);
            } else if now - deadline > TICK * 10 {
                warn!("the server can't keep up, skipping ticks");
                deadline = now;
            }
        }
        info!("all players left, stopping the server");
    }

    fn handle_messages(&mut self, index: usize) {
        while self.clients[index].connected {
            match self.clients[index].connection.try_recv() {
                Ok(Some(message)) => self.handle(index, message),
                Ok(None) => break,
                Err(RecvError::Disconnected) => self.drop_client(index),
                Err(RecvError::Malformed(error)) => self.kick(index, &error.to_string()),
            }
        }
    }

    fn handle(&mut self, index: usize, message: ClientMessage) {
        let greeted = self.clients[index].name.is_some();
        match message {
            ClientMessage::Hello { protocol, name } if !greeted => {
                if protocol != PROTOCOL_VERSION {
                    let reason = format!(
                        "protocol version {protocol} is not supported, the server speaks {PROTOCOL_VERSION}"
                    );
                    return self.kick(index, &reason);
                }
                info!(%name, "player joined");
                let blocks = self
                    .content
                    .blocks()
                    .iter()
                    .map(|(_, def)| def.id.clone())
                    .collect();
                let client = &mut self.clients[index];
                client.name = Some(name);
                client.send(ServerMessage::Welcome {
                    blocks,
                    spawn: self.spawn,
                    bounds: self.config.bounds,
                    time: self.time,
                });
            }
            _ if !greeted => self.kick(index, "expected a hello first"),
            ClientMessage::Hello { .. } => self.kick(index, "said hello twice"),
            ClientMessage::Position(position) => {
                if position.is_finite() {
                    self.clients[index].position = Some(position);
                }
            }
            ClientMessage::ViewDistance(chunks) => {
                self.clients[index].view_distance =
                    i32::from(chunks).clamp(1, self.config.view_distance);
            }
            ClientMessage::LodDistance(blocks) => {
                self.clients[index].lod_distance =
                    i32::from(blocks).min(self.config.max_lod_distance);
            }
            ClientMessage::BreakBlock { pos, seq } => {
                self.block_action(index, pos, BlockId::AIR, seq);
            }
            ClientMessage::PlaceBlock { pos, block, seq } => {
                self.block_action(index, pos, block, seq);
            }
        }
    }

    /// Breaks the block at `pos` (`block` is air) or places `block` there, if
    /// the rules allow it; otherwise tells the client what is really there.
    fn block_action(&mut self, index: usize, pos: BlockPos, block: BlockId, seq: u32) {
        let content = Arc::clone(&self.content);
        let blocks = content.blocks();
        let breaking = block == BlockId::AIR;
        // Players can't place what they couldn't break, and torches need
        // something solid to hold on to.
        let placeable = block != BlockId::UNKNOWN
            && block.index() < blocks.len()
            && blocks.is_visible(block)
            && blocks.is_breakable(block)
            && blocks.support(block).is_none_or(|face| {
                self.world
                    .block(pos.offset(face))
                    .is_some_and(|support| blocks.is_solid(support))
            });
        let in_reach = self.clients[index]
            .position
            .is_some_and(|eye| eye.distance(pos.0.as_dvec3() + 0.5) <= REACH);
        let current = self.world.block(pos);
        // Breaking needs something breakable there, placing needs an empty cell.
        let target_ok = current.is_some_and(|current| {
            if breaking {
                blocks.is_visible(current) && blocks.is_breakable(current)
            } else {
                current == BlockId::AIR
            }
        });
        let allowed =
            (breaking || placeable) && in_reach && self.config.bounds.contains(pos) && target_ok;

        if allowed {
            self.change_block(pos, block);
            // Torches fall when what held them goes.
            if !blocks.is_solid(block) {
                for face in Face::ALL {
                    let near = pos.offset(face);
                    if self
                        .world
                        .block(near)
                        .is_some_and(|near| blocks.support(near) == Some(face.opposite()))
                    {
                        self.change_block(near, BlockId::AIR);
                    }
                }
            }
        } else if let Some(current) = current
            && self.clients[index].sent.contains(&pos.chunk())
        {
            self.clients[index].send(ServerMessage::BlockChanged {
                pos,
                block: current,
            });
        }
        self.clients[index].send(ServerMessage::ActionDone { seq });
    }

    /// Sets a block, updates the light around it and tells every client that
    /// has its chunk.
    fn change_block(&mut self, pos: BlockPos, block: BlockId) {
        let Some(old) = self.world.set_block(pos, block) else {
            return;
        };
        let changed = self.light.block_changed(&mut self.world, pos, old);
        self.light_changed.extend(changed);
        self.modified.insert(pos.chunk());
        for client in &mut self.clients {
            if client.sent.contains(&pos.chunk()) {
                client.send(ServerMessage::BlockChanged { pos, block });
            }
        }
    }

    /// Sends the nearest missing chunks around the client, asks for the ones
    /// that don't exist yet, and tells it to forget those it moved away from.
    fn stream_chunks(&mut self, index: usize) {
        let Some(position) = self.clients[index].position else {
            return;
        };
        let center = BlockPos(position.floor().as_ivec3()).chunk();
        let radius = self.clients[index].view_distance;
        let view = self.view(radius);
        let mut budget = CHUNKS_PER_TICK;
        for offset in view.iter() {
            let pos = ChunkPos(center.0 + *offset);
            if !self.config.bounds.contains_chunk(pos) || self.clients[index].sent.contains(&pos) {
                continue;
            }
            if !self.is_complete(pos) {
                // It and its neighbours need generating or lighting first.
                for near in around(pos) {
                    if self.config.bounds.contains_chunk(near) && self.world.chunk(near).is_none() {
                        self.generate(near);
                    }
                }
                continue;
            }
            if budget == 0 {
                continue;
            }
            let Some(chunk) = self.world.chunk(pos) else {
                continue;
            };
            budget -= 1;
            let message = ServerMessage::Chunk {
                pos,
                chunk: chunk.clone(),
            };
            let client = &mut self.clients[index];
            client.sent.insert(pos);
            client.send(message);
        }

        // One chunk of slack, so walking along a border doesn't make chunks
        // load and unload over and over.
        let (radius, vertical) = (radius + 1, vertical_view_distance(radius) + 1);
        let client = &mut self.clients[index];
        let far: Vec<ChunkPos> = client
            .sent
            .iter()
            .filter(|pos| !in_view(pos.0 - center.0, radius, vertical))
            .copied()
            .collect();
        for pos in far {
            client.sent.remove(&pos);
            client.send(ServerMessage::UnloadChunk(pos));
        }
    }

    fn view(&mut self, radius: i32) -> Arc<[IVec3]> {
        Arc::clone(
            self.views
                .entry(radius)
                .or_insert_with(|| view_offsets(radius, vertical_view_distance(radius)).into()),
        )
    }

    /// Sends the far-away look of the nearest tiles the client doesn't have
    /// yet, asks for the ones that don't exist, and tells it to forget those
    /// it moved away from.
    fn stream_lod(&mut self, index: usize) {
        let Some(position) = self.clients[index].position else {
            return;
        };
        let center = LodTilePos::containing(position.x.floor() as i32, position.z.floor() as i32);
        let reach = if self.lod_unsupported {
            0
        } else {
            self.clients[index].lod_distance
        };
        let radius = (reach + LOD_TILE_SIZE - 1) / LOD_TILE_SIZE;
        let mut budget = LOD_TILES_PER_TICK;
        let wanted = if reach > 0 {
            lod_tiles_around(center, radius)
        } else {
            Vec::new()
        };
        for pos in wanted {
            if self.clients[index].lod_sent.contains(&pos) {
                continue;
            }
            match self.lod_tiles.get(&pos) {
                Some(_) if budget == 0 => {}
                Some(tile) => {
                    budget -= 1;
                    let message = ServerMessage::LodTile {
                        pos,
                        tile: LodTile::clone(tile),
                    };
                    let client = &mut self.clients[index];
                    client.lod_sent.insert(pos);
                    client.send(message);
                }
                None => self.generate_lod(pos),
            }
        }

        // A tile of slack, as with chunks.
        let client = &mut self.clients[index];
        let far: Vec<LodTilePos> = client
            .lod_sent
            .iter()
            .filter(|pos| reach == 0 || !in_lod_reach(**pos, center, radius + 1))
            .copied()
            .collect();
        for pos in far {
            client.lod_sent.remove(&pos);
            client.send(ServerMessage::UnloadLod(pos));
        }
    }

    fn generate_lod(&mut self, pos: LodTilePos) {
        if self.lod_generating.len() >= self.max_generating || !self.lod_generating.insert(pos) {
            return;
        }
        let generator = Arc::clone(&self.generator);
        let done = self.lod_tx.clone();
        rayon::spawn(move || {
            let _ = done.send((pos, LodTile::generate(&*generator, pos)));
        });
    }

    fn generate(&mut self, pos: ChunkPos) {
        if self.generating.len() >= self.max_generating || !self.generating.insert(pos) {
            return;
        }
        let generator = Arc::clone(&self.generator);
        let done = self.generated_tx.clone();
        rayon::spawn(move || {
            // The server may be gone by now; then nobody needs the chunk.
            let _ = done.send((pos, generator.generate(pos)));
        });
    }

    /// Drops generated chunks nobody is near, unless players changed them.
    /// Whole columns stay: light comes down from their top.
    fn unload_unused_chunks(&mut self) {
        // Each player's area, with a margin.
        let areas: Vec<(ChunkPos, i32)> = self
            .clients
            .iter()
            .filter_map(|client| {
                let center = BlockPos(client.position?.floor().as_ivec3()).chunk();
                Some((center, client.view_distance + 2))
            })
            .collect();
        let unused: Vec<ChunkPos> = self
            .world
            .chunks()
            .map(|(pos, _)| pos)
            .filter(|pos| {
                !self.modified.contains(pos)
                    && !areas
                        .iter()
                        .any(|&(center, radius)| in_view(pos.0 - center.0, radius, i32::MAX))
            })
            .collect();
        for pos in unused {
            self.world.remove_chunk(pos);
            self.light.forget(pos);
            self.unlit.remove(&pos);
        }

        // Far-away tiles nobody is near.
        let reaches: Vec<(LodTilePos, i32)> = self
            .clients
            .iter()
            .filter_map(|client| {
                let position = client.position?;
                let center =
                    LodTilePos::containing(position.x.floor() as i32, position.z.floor() as i32);
                let radius = (client.lod_distance + LOD_TILE_SIZE - 1) / LOD_TILE_SIZE;
                Some((center, radius + 2))
            })
            .collect();
        self.lod_tiles.retain(|&pos, _| {
            reaches
                .iter()
                .any(|&(center, radius)| in_lod_reach(pos, center, radius))
        });
    }

    fn kick(&mut self, index: usize, reason: &str) {
        let client = &mut self.clients[index];
        warn!(
            name = client.name.as_deref().unwrap_or("?"),
            reason, "disconnecting a client"
        );
        client.send(ServerMessage::Disconnect {
            reason: reason.to_owned(),
        });
        client.connected = false;
    }

    fn drop_client(&mut self, index: usize) {
        let client = &mut self.clients[index];
        if let Some(name) = &client.name {
            info!(%name, "player left");
        }
        client.connected = false;
    }
}

impl fmt::Debug for Server {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("Server")
            .field("config", &self.config)
            .field("clients", &self.clients.len())
            .field("chunks", &self.world.chunk_count())
            .field("generating", &self.generating.len())
            .field("ticks", &self.ticks)
            .finish_non_exhaustive()
    }
}

/// Starts a single-player server on its own thread and connects one client
/// to it. The server stops when that client disconnects.
pub fn spawn_integrated(
    content: Arc<Content>,
    generator: Arc<dyn Generator>,
    config: ServerConfig,
) -> std::io::Result<(JoinHandle<()>, ClientConnection)> {
    let (client, server_end) = local_pair();
    let mut server = Server::new(content, generator, config);
    server.connect(server_end);
    let thread = thread::Builder::new()
        .name("server".into())
        .spawn(move || server.run_until_empty())?;
    Ok((thread, client))
}

/// Tiles whose middle is within `radius` tiles of the middle of `center`.
fn in_lod_reach(pos: LodTilePos, center: LodTilePos, radius: i32) -> bool {
    let (dx, dz) = (pos.x - center.x, pos.z - center.z);
    dx * dx + dz * dz <= radius * radius
}

/// The tiles within `radius` of `center`, nearest first.
fn lod_tiles_around(center: LodTilePos, radius: i32) -> Vec<LodTilePos> {
    let mut tiles = Vec::new();
    for dz in -radius..=radius {
        for dx in -radius..=radius {
            let pos = LodTilePos::new(center.x + dx, center.z + dz);
            if in_lod_reach(pos, center, radius) {
                tiles.push(pos);
            }
        }
    }
    tiles.sort_by_key(|pos| {
        let (dx, dz) = (pos.x - center.x, pos.z - center.z);
        dx * dx + dz * dz
    });
    tiles
}

/// The chunk and the 26 around it.
fn around(pos: ChunkPos) -> impl Iterator<Item = ChunkPos> {
    (-1..=1).flat_map(move |y| {
        (-1..=1).flat_map(move |z| (-1..=1).map(move |x| ChunkPos(pos.0 + IVec3::new(x, y, z))))
    })
}

fn in_view(offset: IVec3, radius: i32, vertical: i32) -> bool {
    offset.x * offset.x + offset.z * offset.z <= radius * radius && offset.y.abs() <= vertical
}

fn view_offsets(radius: i32, vertical: i32) -> Vec<IVec3> {
    let mut offsets = Vec::new();
    for y in -vertical..=vertical {
        for z in -radius..=radius {
            for x in -radius..=radius {
                let offset = IVec3::new(x, y, z);
                if in_view(offset, radius, vertical) {
                    offsets.push(offset);
                }
            }
        }
    }
    offsets.sort_by_key(|offset| offset.length_squared());
    offsets
}

#[cfg(test)]
mod tests {
    use ruda_core::{
        Appearance, BlockDef, ContentBuilder, CubeTextures, Light, LocalPos, ResourceId,
    };

    use super::*;

    /// Stone below y = 0 on a floor at y = −32, air above.
    struct Flat {
        stone: BlockId,
        floor: BlockId,
    }

    impl Generator for Flat {
        fn generate(&self, pos: ChunkPos) -> Chunk {
            if pos.0.y >= 0 {
                return Chunk::filled(BlockId::AIR);
            }
            let mut chunk = Chunk::filled(self.stone);
            if pos.0.y == -1 {
                for x in 0..32 {
                    for z in 0..32 {
                        chunk.set(LocalPos::new(x, 0, z), self.floor);
                    }
                }
            }
            chunk
        }

        fn surface_height(&self, _x: i32, _z: i32) -> Option<i32> {
            Some(-1)
        }

        fn surface(&self, _x: i32, _z: i32) -> Option<(i32, BlockId)> {
            Some((-1, self.stone))
        }
    }

    struct Harness {
        server: Server,
        client: ClientConnection,
        stone: BlockId,
        bedrock: BlockId,
        torch: BlockId,
    }

    impl Harness {
        fn new() -> Self {
            let mut content = ContentBuilder::new();
            let id: ResourceId = "test:stone".parse().unwrap();
            let stone = content
                .add_block(BlockDef::new(
                    id.clone(),
                    Appearance::Cube(CubeTextures::all(id)),
                ))
                .unwrap();
            let id: ResourceId = "test:bedrock".parse().unwrap();
            let bedrock = content
                .add_block(
                    BlockDef::new(id.clone(), Appearance::Cube(CubeTextures::all(id)))
                        .unbreakable(),
                )
                .unwrap();
            let id: ResourceId = "test:torch".parse().unwrap();
            let torch = content
                .add_block(BlockDef::new(id.clone(), Appearance::Torch { texture: id }))
                .unwrap();
            let config = ServerConfig {
                view_distance: 1,
                // Two chunks tall: y from −32 to 31.
                bounds: WorldBounds {
                    min_y: -32,
                    max_y: 31,
                },
                ..Default::default()
            };
            let mut server = Server::new(
                Arc::new(content.build()),
                Arc::new(Flat {
                    stone,
                    floor: bedrock,
                }),
                config,
            );
            let (client, server_end) = local_pair();
            server.connect(server_end);
            Self {
                server,
                client,
                stone,
                bedrock,
                torch,
            }
        }

        fn send(&self, message: ClientMessage) {
            self.client.send(&message).unwrap();
        }

        /// Ticks until the server sends a message matching `wanted`.
        fn expect(&mut self, wanted: impl Fn(&ServerMessage) -> bool) -> ServerMessage {
            let deadline = Instant::now() + Duration::from_secs(10);
            while Instant::now() < deadline {
                self.server.tick();
                while let Some(message) = self.client.try_recv().unwrap() {
                    if wanted(&message) {
                        return message;
                    }
                }
                thread::sleep(Duration::from_millis(2));
            }
            panic!("the expected message never came");
        }

        fn join(&mut self) {
            self.send(ClientMessage::Hello {
                protocol: PROTOCOL_VERSION,
                name: "tester".into(),
            });
            self.expect(|m| matches!(m, ServerMessage::Welcome { .. }));
            self.send(ClientMessage::Position(DVec3::new(0.5, 1.5, 0.5)));
        }
    }

    #[test]
    fn refuses_other_protocol_versions() {
        let mut harness = Harness::new();
        harness.send(ClientMessage::Hello {
            protocol: PROTOCOL_VERSION + 1,
            name: "old".into(),
        });
        harness.expect(|m| matches!(m, ServerMessage::Disconnect { .. }));
        assert_eq!(harness.server.client_count(), 0);
    }

    #[test]
    fn streams_nearer_chunks_first() {
        // Chunks go out in this order once they are ready.
        let offsets = view_offsets(3, 2);
        assert_eq!(offsets[0], IVec3::ZERO);
        assert!(
            offsets
                .windows(2)
                .all(|pair| pair[0].length_squared() <= pair[1].length_squared())
        );
        assert!(offsets.iter().all(|&offset| in_view(offset, 3, 2)));
    }

    #[test]
    fn applies_valid_actions_and_corrects_rejected_ones() {
        let mut harness = Harness::new();
        harness.join();
        let below = BlockPos::new(0, -1, 0);
        harness.expect(|m| matches!(m, ServerMessage::Chunk { pos, .. } if *pos == below.chunk()));

        harness.send(ClientMessage::BreakBlock { pos: below, seq: 1 });
        let changed = harness.expect(|m| matches!(m, ServerMessage::BlockChanged { .. }));
        assert_eq!(
            changed,
            ServerMessage::BlockChanged {
                pos: below,
                block: BlockId::AIR
            }
        );
        assert_eq!(harness.server.world().block(below), Some(BlockId::AIR));

        // Placing into solid stone is not allowed: the client gets corrected.
        let solid = BlockPos::new(0, -2, 0);
        harness.send(ClientMessage::PlaceBlock {
            pos: solid,
            block: harness.stone,
            seq: 2,
        });
        let correction = harness.expect(|m| matches!(m, ServerMessage::BlockChanged { .. }));
        assert_eq!(
            correction,
            ServerMessage::BlockChanged {
                pos: solid,
                block: harness.stone
            }
        );
        harness.expect(|m| matches!(m, ServerMessage::ActionDone { seq: 2 }));

        // Out of reach.
        let far = BlockPos::new(0, -1, 20);
        harness.send(ClientMessage::BreakBlock { pos: far, seq: 3 });
        harness.expect(|m| matches!(m, ServerMessage::ActionDone { seq: 3 }));
        assert_ne!(harness.server.world().block(far), Some(BlockId::AIR));
    }

    #[test]
    fn keeps_unbreakable_blocks_and_world_bounds() {
        let mut harness = Harness::new();
        harness.join();
        harness.send(ClientMessage::Position(DVec3::new(0.5, -29.5, 0.5)));
        let floor = BlockPos::new(0, -32, 0);
        harness.expect(|m| matches!(m, ServerMessage::Chunk { pos, .. } if *pos == floor.chunk()));

        harness.send(ClientMessage::BreakBlock { pos: floor, seq: 1 });
        let correction = harness.expect(|m| matches!(m, ServerMessage::BlockChanged { .. }));
        assert_eq!(
            correction,
            ServerMessage::BlockChanged {
                pos: floor,
                block: harness.bedrock
            }
        );
        assert_eq!(harness.server.world().block(floor), Some(harness.bedrock));

        // Players can't place unbreakable blocks either.
        let open = BlockPos::new(1, -29, 0);
        harness.send(ClientMessage::BreakBlock { pos: open, seq: 2 });
        harness.expect(|m| matches!(m, ServerMessage::ActionDone { seq: 2 }));
        harness.send(ClientMessage::PlaceBlock {
            pos: open,
            block: harness.bedrock,
            seq: 3,
        });
        harness.expect(|m| matches!(m, ServerMessage::ActionDone { seq: 3 }));
        assert_eq!(harness.server.world().block(open), Some(BlockId::AIR));
    }

    #[test]
    fn sends_chunks_lit_and_their_light_when_it_changes() {
        let mut harness = Harness::new();
        harness.join();
        let sky = ChunkPos::new(0, 0, 0);
        let ServerMessage::Chunk { chunk, .. } =
            harness.expect(|m| matches!(m, ServerMessage::Chunk { pos, .. } if *pos == sky))
        else {
            unreachable!()
        };
        assert_eq!(chunk.light().get(LocalPos::new(5, 0, 5)), Light::SKY);

        // A roof over (5, 2, 5) shades it: sky light only comes in sideways.
        harness.send(ClientMessage::PlaceBlock {
            pos: BlockPos::new(5, 3, 5),
            block: harness.stone,
            seq: 1,
        });
        let ServerMessage::Light { light, .. } =
            harness.expect(|m| matches!(m, ServerMessage::Light { pos, .. } if *pos == sky))
        else {
            unreachable!()
        };
        assert_eq!(light.get(LocalPos::new(5, 2, 5)).sky(), 14);
        assert_eq!(light.get(LocalPos::new(6, 2, 5)), Light::SKY);
    }

    #[test]
    fn keeps_the_time_of_day() {
        let mut harness = Harness::new();
        harness.send(ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            name: "tester".into(),
        });
        let ServerMessage::Welcome { time, .. } =
            harness.expect(|m| matches!(m, ServerMessage::Welcome { .. }))
        else {
            unreachable!()
        };
        let ServerMessage::Time(later) = harness.expect(|m| matches!(m, ServerMessage::Time(_)))
        else {
            unreachable!()
        };
        assert!(later > time);
    }

    #[test]
    fn torches_need_and_fall_with_their_support() {
        let mut harness = Harness::new();
        harness.join();
        harness.expect(
            |m| matches!(m, ServerMessage::Chunk { pos, .. } if *pos == ChunkPos::new(0, 0, 0)),
        );
        let blocks = harness.server.content.blocks().clone();
        let on_floor = blocks.placed(harness.torch, Face::PosY).unwrap();
        let on_wall = blocks.placed(harness.torch, Face::NegZ).unwrap();

        // Not in mid-air.
        harness.send(ClientMessage::PlaceBlock {
            pos: BlockPos::new(3, 3, 3),
            block: on_floor,
            seq: 1,
        });
        harness.expect(|m| matches!(m, ServerMessage::ActionDone { seq: 1 }));
        assert_eq!(
            harness.server.world().block(BlockPos::new(3, 3, 3)),
            Some(BlockId::AIR)
        );

        // On the ground, and on the side of a block.
        let wall = BlockPos::new(3, 0, 5);
        let hanging = BlockPos::new(3, 0, 4);
        for (seq, pos, block) in [
            (2, BlockPos::new(3, 0, 3), on_floor),
            (3, wall, harness.stone),
            (4, hanging, on_wall),
        ] {
            harness.send(ClientMessage::PlaceBlock { pos, block, seq });
            harness.expect(|m| matches!(m, ServerMessage::ActionDone { seq: s } if *s == seq));
            assert_eq!(harness.server.world().block(pos), Some(block));
        }

        // Without its wall, the torch falls.
        harness.send(ClientMessage::BreakBlock { pos: wall, seq: 5 });
        harness.expect(|m| {
            matches!(m, ServerMessage::BlockChanged { pos, block } if *pos == hanging && *block == BlockId::AIR)
        });
        assert_eq!(
            harness.server.world().block(BlockPos::new(3, 0, 3)),
            Some(on_floor)
        );
    }

    #[test]
    fn sends_the_far_away_look_and_takes_it_back() {
        let mut harness = Harness::new();
        harness.join();
        harness.send(ClientMessage::LodDistance(300));
        let ServerMessage::LodTile { pos, tile } =
            harness.expect(|m| matches!(m, ServerMessage::LodTile { .. }))
        else {
            unreachable!()
        };
        assert_eq!(pos, LodTilePos::new(0, 0));
        assert!(tile.is_complete());
        assert_eq!(tile.cell(3, 3), (-1, harness.stone));

        harness.send(ClientMessage::LodDistance(0));
        harness.expect(|m| matches!(m, ServerMessage::UnloadLod(_)));
    }

    /// Waits for `count` chunks, then for a few more ticks in case there are
    /// more than that.
    fn received_chunks(harness: &mut Harness, count: usize) -> HashSet<ChunkPos> {
        let mut received = HashSet::new();
        while received.len() < count {
            if let ServerMessage::Chunk { pos, .. } =
                harness.expect(|m| matches!(m, ServerMessage::Chunk { .. }))
            {
                received.insert(pos);
            }
        }
        for _ in 0..5 {
            harness.server.tick();
        }
        while let Some(message) = harness.client.try_recv().unwrap() {
            if let ServerMessage::Chunk { pos, .. } = message {
                received.insert(pos);
            }
        }
        received
    }

    #[test]
    fn sends_no_chunks_outside_the_world() {
        let mut harness = Harness::new();
        harness.join();
        // Radius 1 around the origin is 5 columns, two chunks tall in bounds.
        let received = received_chunks(&mut harness, 10);
        assert_eq!(received.len(), 10);
        assert!(received.iter().all(|pos| (-1..=0).contains(&pos.0.y)));
    }

    #[test]
    fn caps_the_view_distance_players_ask_for() {
        let mut harness = Harness::new();
        harness.join();
        harness.send(ClientMessage::ViewDistance(12));
        assert_eq!(received_chunks(&mut harness, 10).len(), 10);
    }
}
