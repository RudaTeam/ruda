//! A client and a server with the base game, connected in memory.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::{DVec3, Vec2};
use ruda_client::Client;
use ruda_core::{BlockId, ChunkPos, ContentBuilder, Light, WorldBounds};
use ruda_protocol::PlayerInput;
use ruda_server::{Server, ServerConfig};
use ruda_world::{RayHit, raycast};

struct Game {
    server: Server,
    client: Client,
}

impl Game {
    fn new() -> Self {
        let mut content = ContentBuilder::new();
        ruda_base::register(&mut content).unwrap();
        let content = Arc::new(content.build());
        let generator =
            Arc::new(ruda_base::terrain(content.blocks(), 2024, WorldBounds::DEFAULT).unwrap());
        let config = ServerConfig {
            view_distance: 2,
            ..Default::default()
        };
        let mut server = Server::new(content.clone(), generator, config);
        let (connection, server_end) = ruda_net::local_pair();
        server.connect(server_end);
        let client = Client::connect(connection, content, "tester").unwrap();
        Self { server, client }
    }

    /// Runs both sides until `done` holds.
    fn run_until(&mut self, mut done: impl FnMut(&Client) -> bool) {
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            self.server.tick();
            for _ in self.client.update() {}
            if done(&self.client) {
                return;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out");
    }

    /// Joins and waits for the world around the player: the chunks its feet
    /// are in, below them and to the side.
    fn join(&mut self) -> DVec3 {
        self.run_until(|client| client.player().is_some());
        let feet = self.client.player().unwrap().position;
        let chunk = ruda_core::BlockPos(feet.floor().as_ivec3()).chunk().0;
        self.run_until(|client| {
            (-1..=0).all(|y| {
                (-1..=1).all(|z| {
                    (-1..=1).all(|x| {
                        let pos = ChunkPos(chunk + glam::IVec3::new(x, y, z));
                        client.world().chunk(pos).is_some()
                    })
                })
            })
        });
        feet
    }
}

fn ground_below(client: &Client, from: DVec3) -> Option<RayHit> {
    raycast(from, DVec3::NEG_Y, 8.0, |pos| client.is_solid(pos))
}

#[test]
fn joins_receives_terrain_and_edits_it() {
    let mut game = Game::new();
    let feet = game.join();
    // The player starts standing on the ground.
    let ground = ground_below(&game.client, feet + DVec3::Y * 0.5).unwrap();
    assert!(ground.distance < 0.5 + 1e-9);

    assert!(game.client.break_block(ground.block));
    // The hole is lit right away, not only once the server's light comes.
    let light = |client: &Client| {
        let chunk = client.world().chunk(ground.block.chunk()).unwrap();
        chunk.light().get(ground.block.local())
    };
    assert_eq!(light(&game.client).sky(), Light::MAX);
    game.run_until(|client| client.pending_actions() == 0);
    assert_eq!(light(&game.client).sky(), Light::MAX);
    assert_eq!(game.server.world().block(ground.block), Some(BlockId::AIR));
    assert_eq!(game.client.world().block(ground.block), Some(BlockId::AIR));

    let planks = game
        .client
        .content()
        .blocks()
        .id(&ruda_base::id("planks").unwrap())
        .unwrap();
    assert!(game.client.place_block(ground.block, planks));
    game.run_until(|client| client.pending_actions() == 0);
    assert_eq!(game.server.world().block(ground.block), Some(planks));

    // Not into the player, though.
    let feet_block = ground.block.offset(ruda_core::Face::PosY);
    assert!(!game.client.place_block(feet_block, planks));
}

#[test]
fn predicts_where_the_server_moves_the_player() {
    let mut game = Game::new();
    let start = game.join();
    for _ in 0..15 {
        game.client.tick_player(PlayerInput {
            walk: Vec2::new(0.3, 1.0),
            jump: true,
            ..Default::default()
        });
        game.server.tick();
        for _ in game.client.update() {}
    }
    let predicted = *game.client.player().unwrap();
    game.run_until(|client| client.pending_inputs() == 0);
    let decided = *game.client.player().unwrap();
    assert!(start.distance(decided.position) > 1.0);
    assert_eq!(predicted, decided);
    assert_eq!(game.client.player_position(1.0), Some(decided.position));
}
