//! The client's side of the game: the world as the server has sent it, and
//! the player's actions on it.
//!
//! Actions apply locally right away so the game feels instant; the server
//! either confirms them or sends back what the world really looks like.

use std::sync::Arc;

use glam::DVec3;
use ruda_core::{BlockId, BlockPos, ChunkPos, Content, WorldBounds};
use ruda_net::{ClientConnection, Disconnected, RecvError};
use ruda_protocol::{ClientMessage, PROTOCOL_VERSION, ServerMessage};
use ruda_world::World;

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
    /// The connection is over; no events follow.
    Disconnected {
        reason: String,
    },
}

#[derive(Debug)]
pub struct Client {
    connection: ClientConnection,
    content: Arc<Content>,
    world: World,
    spawn: Option<DVec3>,
    bounds: Option<WorldBounds>,
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
            spawn: None,
            bounds: None,
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
            } => {
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

    /// Breaks a solid block. Returns false if there is nothing that can be
    /// broken.
    pub fn break_block(&mut self, pos: BlockPos) -> bool {
        let breakable = self
            .world
            .block(pos)
            .is_some_and(|block| self.content.blocks().is_breakable(block));
        if !self.is_solid(pos) || !breakable {
            return false;
        }
        let seq = self.next_seq();
        self.act(pos, BlockId::AIR, ClientMessage::BreakBlock { pos, seq });
        true
    }

    /// Places `block` into an empty cell. Returns false if that isn't possible.
    pub fn place_block(&mut self, pos: BlockPos, block: BlockId) -> bool {
        let blocks = self.content.blocks();
        let empty = self.world.block(pos).is_some() && !self.is_solid(pos);
        let inside = self.bounds.is_some_and(|bounds| bounds.contains(pos));
        let placeable =
            block != BlockId::UNKNOWN && blocks.is_solid(block) && blocks.is_breakable(block);
        if !empty || !inside || !placeable {
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
