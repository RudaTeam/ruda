//! A client and a server with the base game, connected in memory.

use std::sync::Arc;
use std::time::{Duration, Instant};

use glam::DVec3;
use ruda_client::{Client, Event};
use ruda_core::{BlockId, ContentBuilder, WorldBounds};
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

    /// Runs both sides until `done` holds, collecting the client's events.
    fn run_until(&mut self, mut done: impl FnMut(&Client, &[Event]) -> bool) -> Vec<Event> {
        let mut events = Vec::new();
        let deadline = Instant::now() + Duration::from_secs(20);
        while Instant::now() < deadline {
            self.server.tick();
            events.extend(self.client.update());
            if done(&self.client, &events) {
                return events;
            }
            std::thread::sleep(Duration::from_millis(2));
        }
        panic!("timed out; events so far: {events:?}");
    }
}

fn ground_below(client: &Client, from: DVec3) -> Option<RayHit> {
    raycast(from, DVec3::NEG_Y, 8.0, |pos| client.is_solid(pos))
}

#[test]
fn joins_receives_terrain_and_edits_it() {
    let mut game = Game::new();
    let events =
        game.run_until(|_, events| events.iter().any(|e| matches!(e, Event::Joined { .. })));
    let Some(Event::Joined { spawn }) = events
        .into_iter()
        .find(|e| matches!(e, Event::Joined { .. }))
    else {
        unreachable!()
    };
    game.client.set_position(spawn);

    // Wait for the ground under the spawn point.
    game.run_until(|client, _| ground_below(client, spawn).is_some());
    let ground = ground_below(&game.client, spawn).unwrap();

    assert!(game.client.break_block(ground.block));
    game.run_until(|client, _| client.pending_actions() == 0);
    assert_eq!(game.server.world().block(ground.block), Some(BlockId::AIR));
    assert_eq!(game.client.world().block(ground.block), Some(BlockId::AIR));

    let planks = game
        .client
        .content()
        .blocks()
        .id(&ruda_base::id("planks").unwrap())
        .unwrap();
    assert!(game.client.place_block(ground.block, planks));
    game.run_until(|client, _| client.pending_actions() == 0);
    assert_eq!(game.server.world().block(ground.block), Some(planks));
}
