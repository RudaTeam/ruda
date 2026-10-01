//! Messages between the client and the server.
//!
//! Both directions are plain enums serialized with postcard. Single-player
//! passes them through an in-memory channel, multiplayer over the network,
//! and either way the client learns about the world only from these.

use ruda_core::{BlockId, BlockPos, ChunkPos, ResourceId, WorldBounds};
pub use ruda_sim::{PlayerInput, PlayerState};
use ruda_world::lod::{LodTile, LodTilePos};
use ruda_world::{Chunk, ChunkLight};
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

/// Bumped on every incompatible change to the messages.
pub const PROTOCOL_VERSION: u32 = 7;

/// Simulation steps per second.
pub const TICK_RATE: u32 = 20;

/// Ticks in a day: 20 minutes at 20 ticks a second. Time 0 is sunrise,
/// a quarter of the day is noon, half is sunset and three quarters midnight.
pub const DAY_LENGTH: u64 = 24_000;

/// How far a player can reach to break or place blocks, measured in blocks
/// from the eyes to the block's centre. Clients aim within it and the server
/// rejects actions beyond it.
pub const REACH: f64 = 8.0;

/// Sent by the client.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ClientMessage {
    /// The first message on a new connection.
    Hello { protocol: u32, name: String },
    /// What the player wants to do this tick; sent every tick. The server
    /// moves the player by it and answers with [`ServerMessage::Player`].
    Input(PlayerInput),
    /// How far, in chunks, the client wants the world around it. The server
    /// may send less.
    ViewDistance(u8),
    /// How far, in blocks, the client wants the far-away look of the world;
    /// 0 for none. The server may send less.
    LodDistance(u16),
    /// Asks to break the block at `pos`. The server answers with
    /// [`ServerMessage::ActionDone`] carrying the same `seq`.
    BreakBlock { pos: BlockPos, seq: u32 },
    /// Asks to place `block` at `pos`, which has to be empty.
    PlaceBlock {
        pos: BlockPos,
        block: BlockId,
        seq: u32,
    },
}

/// Sent by the server.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum ServerMessage {
    /// Accepts the connection. Block id `n` of this session is `blocks[n]`.
    Welcome {
        blocks: Vec<ResourceId>,
        bounds: WorldBounds,
        /// Ticks since the world began, see [`DAY_LENGTH`].
        time: u64,
        /// Decides the clouds; derived from the world's seed, which it
        /// doesn't give away.
        sky_seed: u64,
    },
    /// Closes the connection.
    Disconnect { reason: String },
    /// Where this client's player is and how it moves, once the server has
    /// applied its inputs up to `ack`. Also puts the player somewhere: where
    /// it starts, or where the server moved it.
    Player { ack: u32, state: PlayerState },
    /// A chunk with its light. The server sends chunks only once their
    /// light is final; later it changes only with [`ServerMessage::Light`].
    Chunk { pos: ChunkPos, chunk: Chunk },
    /// New light for a chunk the client has, after blocks changed nearby.
    Light { pos: ChunkPos, light: ChunkLight },
    /// The world's time, sent now and then so clocks don't drift apart.
    Time(u64),
    /// The far-away look of a tile of the world.
    LodTile { pos: LodTilePos, tile: LodTile },
    /// The client should forget this tile's far-away look.
    UnloadLod(LodTilePos),
    /// The client should forget this chunk.
    UnloadChunk(ChunkPos),
    /// The block at `pos` is now `block`: someone changed it, or the server
    /// corrects an action of this client that it rejected.
    BlockChanged { pos: BlockPos, block: BlockId },
    /// The server has handled action `seq`; anything it changed has already
    /// been sent as [`ServerMessage::BlockChanged`].
    ActionDone { seq: u32 },
}

pub fn encode(message: &impl Serialize) -> Vec<u8> {
    postcard::to_allocvec(message).expect("protocol messages always serialize")
}

pub fn decode<T: DeserializeOwned>(bytes: &[u8]) -> Result<T, DecodeError> {
    Ok(postcard::from_bytes(bytes)?)
}

#[derive(Debug, thiserror::Error)]
#[error("malformed message: {0}")]
pub struct DecodeError(#[from] postcard::Error);

#[cfg(test)]
mod tests {
    use ruda_core::LocalPos;

    use super::*;

    fn round_trip<T: Serialize + DeserializeOwned + PartialEq + std::fmt::Debug>(message: T) {
        assert_eq!(decode::<T>(&encode(&message)).unwrap(), message);
    }

    #[test]
    fn messages_round_trip() {
        round_trip(ClientMessage::Hello {
            protocol: PROTOCOL_VERSION,
            name: "player".into(),
        });
        round_trip(ClientMessage::Input(PlayerInput {
            seq: 3,
            walk: glam::Vec2::new(0.5, -1.0),
            yaw: 1.5,
            jump: true,
            ..Default::default()
        }));
        round_trip(ClientMessage::PlaceBlock {
            pos: BlockPos::new(-1, 2, -3),
            block: BlockId::from_raw(7),
            seq: 42,
        });

        let mut chunk = Chunk::filled(BlockId::AIR);
        chunk.set(LocalPos::new(1, 2, 3), BlockId::UNKNOWN);
        round_trip(ServerMessage::Chunk {
            pos: ChunkPos::new(4, -5, 6),
            chunk,
        });
        round_trip(ServerMessage::Welcome {
            blocks: vec!["ruda:air".parse().unwrap(), "base:stone".parse().unwrap()],
            bounds: WorldBounds::DEFAULT,
            time: 1234,
            sky_seed: 99,
        });
        round_trip(ServerMessage::Light {
            pos: ChunkPos::new(1, 2, 3),
            light: ChunkLight::uniform(ruda_core::Light::SKY),
        });
        round_trip(ServerMessage::Time(99));
        round_trip(ServerMessage::Player {
            ack: 7,
            state: PlayerState::flying(glam::DVec3::new(0.5, 70.0, -1e9)),
        });
        round_trip(ServerMessage::LodTile {
            pos: LodTilePos::new(-3, 4),
            tile: LodTile {
                heights: vec![-5, 120],
                blocks: vec![BlockId::from_raw(4), BlockId::AIR],
            },
        });
    }

    #[test]
    fn rejects_invalid_resource_ids() {
        let message = ServerMessage::Welcome {
            blocks: vec!["base:stone".parse().unwrap()],
            bounds: WorldBounds::DEFAULT,
            time: 0,
            sky_seed: 0,
        };
        let mut bytes = encode(&message);
        let at = bytes.windows(5).position(|w| w == b"stone").unwrap();
        bytes[at] = b'S';
        assert!(decode::<ServerMessage>(&bytes).is_err());
    }
}
