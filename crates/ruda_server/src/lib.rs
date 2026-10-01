//! The authoritative game server.
//!
//! The server owns the world: clients only ask for changes and the server
//! decides. Single-player runs it on a thread next to the client, the
//! dedicated server on its own.

use std::collections::HashSet;
use std::fmt;
use std::sync::Arc;
use std::sync::mpsc::{self, Receiver, Sender};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use glam::{DVec3, IVec3};
use ruda_core::{BlockId, BlockPos, ChunkPos, Content};
use ruda_net::{ClientConnection, RecvError, ServerConnection, local_pair};
use ruda_protocol::{ClientMessage, PROTOCOL_VERSION, REACH, ServerMessage};
use ruda_world::{Chunk, Generator, World};
use tracing::{info, warn};

/// Simulation steps per second.
pub const TICK_RATE: u32 = 20;
const TICK: Duration = Duration::from_millis(1000 / TICK_RATE as u64);

/// Most chunks sent to one client in a tick.
const CHUNKS_PER_TICK: usize = 64;

#[derive(Clone, Copy, Debug)]
pub struct ServerConfig {
    /// Radius, in chunks, of the area streamed around each player.
    pub view_distance: i32,
    /// Vertical radius in chunks: worlds are far wider than they are tall.
    pub vertical_view_distance: i32,
}

impl Default for ServerConfig {
    fn default() -> Self {
        Self {
            view_distance: 6,
            vertical_view_distance: 3,
        }
    }
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
    clients: Vec<RemoteClient>,
    /// Offsets of the chunks streamed around a player, nearest first.
    view: Vec<IVec3>,
    spawn: DVec3,
    ticks: u64,
}

struct RemoteClient {
    connection: ServerConnection,
    /// Set once the client has said hello.
    name: Option<String>,
    position: Option<DVec3>,
    sent: HashSet<ChunkPos>,
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
        Self {
            view: view_offsets(config.view_distance, config.vertical_view_distance),
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
            sent: HashSet::new(),
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
        }
        for index in 0..self.clients.len() {
            self.handle_messages(index);
        }
        self.clients.retain(|client| client.connected);
        for index in 0..self.clients.len() {
            self.stream_chunks(index);
        }
        self.ticks += 1;
        if self.ticks.is_multiple_of(u64::from(TICK_RATE)) {
            self.unload_unused_chunks();
        }
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
                });
            }
            _ if !greeted => self.kick(index, "expected a hello first"),
            ClientMessage::Hello { .. } => self.kick(index, "said hello twice"),
            ClientMessage::Position(position) => {
                if position.is_finite() {
                    self.clients[index].position = Some(position);
                }
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
        let placeable =
            block != BlockId::UNKNOWN && block.index() < blocks.len() && blocks.is_solid(block);
        let in_reach = self.clients[index]
            .position
            .is_some_and(|eye| eye.distance(pos.0.as_dvec3() + 0.5) <= REACH);
        let current = self.world.block(pos);
        // Breaking needs something solid there, placing needs an empty cell.
        let allowed = (breaking || placeable)
            && in_reach
            && current.is_some_and(|current| blocks.is_solid(current) == breaking);

        if allowed {
            self.world.set_block(pos, block);
            self.modified.insert(pos.chunk());
            for client in &mut self.clients {
                if client.sent.contains(&pos.chunk()) {
                    client.send(ServerMessage::BlockChanged { pos, block });
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

    /// Sends the nearest missing chunks around the client, asks for the ones
    /// that don't exist yet, and tells it to forget those it moved away from.
    fn stream_chunks(&mut self, index: usize) {
        let Some(position) = self.clients[index].position else {
            return;
        };
        let center = BlockPos(position.floor().as_ivec3()).chunk();
        let mut budget = CHUNKS_PER_TICK;
        for i in 0..self.view.len() {
            let pos = ChunkPos(center.0 + self.view[i]);
            if self.clients[index].sent.contains(&pos) {
                continue;
            }
            match self.world.chunk(pos) {
                Some(_) if budget == 0 => {}
                Some(chunk) => {
                    budget -= 1;
                    let message = ServerMessage::Chunk {
                        pos,
                        chunk: chunk.clone(),
                    };
                    let client = &mut self.clients[index];
                    client.sent.insert(pos);
                    client.send(message);
                }
                None => self.generate(pos),
            }
        }

        // One chunk of slack, so walking along a border doesn't make chunks
        // load and unload over and over.
        let (radius, vertical) = (
            self.config.view_distance + 1,
            self.config.vertical_view_distance + 1,
        );
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
    fn unload_unused_chunks(&mut self) {
        let centers: Vec<ChunkPos> = self
            .clients
            .iter()
            .filter_map(|client| client.position)
            .map(|position| BlockPos(position.floor().as_ivec3()).chunk())
            .collect();
        let (radius, vertical) = (
            self.config.view_distance + 2,
            self.config.vertical_view_distance + 2,
        );
        let unused: Vec<ChunkPos> = self
            .world
            .chunks()
            .map(|(pos, _)| pos)
            .filter(|pos| {
                !self.modified.contains(pos)
                    && !centers
                        .iter()
                        .any(|center| in_view(pos.0 - center.0, radius, vertical))
            })
            .collect();
        for pos in unused {
            self.world.remove_chunk(pos);
        }
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
    use ruda_core::{Appearance, BlockDef, ContentBuilder, CubeTextures, ResourceId};

    use super::*;

    /// Stone below y = 0, air above.
    struct Flat(BlockId);

    impl Generator for Flat {
        fn generate(&self, pos: ChunkPos) -> Chunk {
            Chunk::filled(if pos.0.y < 0 { self.0 } else { BlockId::AIR })
        }

        fn surface_height(&self, _x: i32, _z: i32) -> Option<i32> {
            Some(-1)
        }
    }

    struct Harness {
        server: Server,
        client: ClientConnection,
        stone: BlockId,
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
            let config = ServerConfig {
                view_distance: 1,
                vertical_view_distance: 1,
            };
            let mut server = Server::new(Arc::new(content.build()), Arc::new(Flat(stone)), config);
            let (client, server_end) = local_pair();
            server.connect(server_end);
            Self {
                server,
                client,
                stone,
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
    fn streams_the_nearest_chunk_first() {
        let mut harness = Harness::new();
        harness.join();
        let first = harness.expect(|m| matches!(m, ServerMessage::Chunk { .. }));
        assert!(matches!(first, ServerMessage::Chunk { pos, .. } if pos == ChunkPos::new(0, 0, 0)));
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
}
