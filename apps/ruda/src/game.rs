//! The game itself: an integrated server, the client talking to it, and the
//! player's camera and controls.

use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use glam::{DVec3, IVec3, Vec3};
use ruda_client::{Client, Event};
use ruda_core::{BlockId, BlockPos, CHUNK_SIZE, ContentBuilder, Light, WorldBounds};
use ruda_input::{Action, Input};
use ruda_protocol::{DAY_LENGTH, REACH};
use ruda_render::{Camera, ChunkMesher, Renderer, Scene};
use ruda_server::ServerConfig;
use ruda_world::{RayHit, raycast};
use tracing::{info, warn};

/// Flying speed in blocks per second.
const SPEED: f64 = 12.0;
const SPRINT_SPEED: f64 = 40.0;
/// Radians of camera turn per unit of mouse movement.
const MOUSE_SENSITIVITY: f32 = 0.0025;
/// See [`Game::is_loaded`].
const LOAD_QUIET: Duration = Duration::from_secs(1);
/// The farthest the integrated server streams the world, in chunks.
pub const MAX_VIEW_DISTANCE: u8 = 32;

#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    pub seed: u64,
    /// In chunks.
    pub view_distance: u8,
    /// Vertical field of view in degrees.
    pub fov: f32,
    /// Where the camera starts instead of the spawn point.
    pub camera: Option<CameraStart>,
    /// The time the world starts at, in ticks; see [`DAY_LENGTH`].
    pub time: Option<u64>,
}

/// A camera position and direction, angles in degrees.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CameraStart {
    pub position: DVec3,
    pub yaw: f32,
    pub pitch: f32,
}

impl std::str::FromStr for CameraStart {
    type Err = String;

    /// `x,y,z` or `x,y,z,yaw,pitch`.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let numbers = text
            .split(',')
            .map(|part| part.trim().parse::<f64>())
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| error.to_string())?;
        let (position, yaw, pitch) = match numbers[..] {
            [x, y, z] => (DVec3::new(x, y, z), 0.0, 0.0),
            [x, y, z, yaw, pitch] => (DVec3::new(x, y, z), yaw, pitch),
            _ => return Err("expected x,y,z or x,y,z,yaw,pitch".into()),
        };
        Ok(Self {
            position,
            yaw: yaw as f32,
            pitch: pitch as f32,
        })
    }
}

/// What the window should do after an update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Pause,
}

#[derive(Debug)]
pub struct Game {
    client: Client,
    server: Option<JoinHandle<()>>,
    mesher: ChunkMesher,
    pub input: Input,
    camera: Camera,
    hotbar: Vec<(&'static str, BlockId)>,
    selected: usize,
    target: Option<RayHit>,
    view_distance: f32,
    joined: bool,
    start: Option<CameraStart>,
    /// When chunks last arrived, left or got new geometry.
    last_change: Instant,
}

impl Game {
    pub fn start(config: GameConfig, renderer: &mut Renderer) -> Result<Self> {
        let mut content = ContentBuilder::new();
        ruda_base::register(&mut content)?;
        let content = Arc::new(content.build());
        let bounds = WorldBounds::DEFAULT;
        let generator = Arc::new(ruda_base::terrain(content.blocks(), config.seed, bounds)?);
        let mut server_config = ServerConfig {
            view_distance: i32::from(MAX_VIEW_DISTANCE),
            bounds,
            ..Default::default()
        };
        if let Some(time) = config.time {
            server_config.start_time = time;
        }
        let (server, connection) =
            ruda_server::spawn_integrated(Arc::clone(&content), generator, server_config)
                .context("failed to start the server")?;
        let mut client = Client::connect(connection, Arc::clone(&content), "player")
            .map_err(|_| anyhow!("the server stopped before the game started"))?;
        client.set_view_distance(config.view_distance);

        let faces = Arc::new(renderer.load_block_textures(&content));
        renderer.set_sky_textures(ruda_base::SUN, ruda_base::MOON);
        let hotbar = ruda_base::HOTBAR
            .iter()
            .filter_map(|&name| Some((name, content.blocks().id(&ruda_base::id(name).ok()?)?)))
            .collect();
        info!(seed = config.seed, "world started");
        let mut camera = Camera::new(DVec3::new(0.5, 100.0, 0.5));
        camera.fov_y = config.fov.to_radians();
        Ok(Self {
            client,
            server: Some(server),
            mesher: ChunkMesher::new(faces),
            input: Input::default(),
            camera,
            hotbar,
            selected: 0,
            target: None,
            view_distance: (i32::from(config.view_distance) * CHUNK_SIZE) as f32,
            joined: false,
            start: config.camera,
            last_change: Instant::now(),
        })
    }

    /// Advances the game by `dt` seconds. Mouse movement only turns the
    /// camera and clicks only act while the cursor is grabbed.
    pub fn update(
        &mut self,
        dt: f64,
        cursor_grabbed: bool,
        renderer: &mut Renderer,
    ) -> Result<Control> {
        for event in self.client.update() {
            if matches!(event, Event::ChunkLoaded(_) | Event::ChunkUnloaded(_)) {
                self.last_change = Instant::now();
            }
            match event {
                Event::Joined { spawn } => {
                    match self.start {
                        Some(start) => {
                            self.camera.position = start.position;
                            self.camera.yaw = 0.0;
                            self.camera.pitch = 0.0;
                            self.camera
                                .rotate(start.yaw.to_radians(), start.pitch.to_radians());
                        }
                        None => {
                            self.camera.position = spawn;
                            self.camera.pitch = -0.3;
                        }
                    }
                    self.joined = true;
                }
                Event::ChunkLoaded(pos) => self.mesher.chunk_loaded(pos),
                Event::ChunkUnloaded(pos) => {
                    self.mesher.forget(pos);
                    renderer.remove_chunk(pos);
                }
                Event::BlockChanged(pos) => {
                    // A block next to a chunk border also changes the
                    // neighbour's faces and the shading of their corners.
                    for offset in (-1..=1).flat_map(|y| {
                        (-1..=1).flat_map(move |z| (-1..=1).map(move |x| IVec3::new(x, y, z)))
                    }) {
                        self.mesher.mark_dirty(BlockPos(pos.0 + offset).chunk());
                    }
                }
                Event::LightChanged(chunks) => {
                    for pos in chunks {
                        self.mesher.mark_dirty(pos);
                    }
                }
                Event::Disconnected { reason } => return Err(anyhow!("disconnected: {reason}")),
            }
        }

        let look = self.input.take_look();
        if cursor_grabbed {
            self.camera
                .rotate(look.x * MOUSE_SENSITIVITY, -look.y * MOUSE_SENSITIVITY);
        }
        self.fly(dt);
        if self.joined {
            self.client.set_position(self.camera.position);
        }

        // Aim a little short of the reach limit: the server measures to the
        // block's centre, the ray to its nearest face.
        let client = &self.client;
        self.target = raycast(
            self.camera.position,
            self.camera.forward().as_dvec3(),
            REACH - 1.0,
            |pos| client.is_solid(pos),
        );

        let mut control = Control::Continue;
        for action in self.input.take_pressed() {
            match action {
                Action::Break if cursor_grabbed => {
                    if let Some(hit) = self.target {
                        self.client.break_block(hit.block);
                    }
                }
                Action::Place if cursor_grabbed => self.place(),
                Action::Hotbar(slot) if usize::from(slot) < self.hotbar.len() => {
                    self.selected = usize::from(slot);
                }
                Action::Pause => control = Control::Pause,
                _ => {}
            }
        }

        let center = BlockPos(self.camera.position.floor().as_ivec3()).chunk();
        self.mesher.schedule(self.client.world(), center);
        for (pos, mesh) in self.mesher.finished() {
            renderer.upload_chunk(pos, &mesh);
            self.last_change = Instant::now();
        }
        Ok(control)
    }

    /// Free flight, horizontally along the view and straight up or down.
    fn fly(&mut self, dt: f64) {
        let movement = self.input.movement();
        if movement == Vec3::ZERO {
            return;
        }
        let forward = self.camera.forward();
        let forward = Vec3::new(forward.x, 0.0, forward.z).normalize_or_zero();
        let direction =
            (forward * movement.z + self.camera.right() * movement.x + Vec3::Y * movement.y)
                .normalize_or_zero();
        let speed = if self.input.is_held(Action::Sprint) {
            SPRINT_SPEED
        } else {
            SPEED
        };
        self.camera.position += direction.as_dvec3() * speed * dt;
        // Stay above the floor of the world; above the build limit is fine
        // for a look around.
        if let Some(bounds) = self.client.bounds() {
            self.camera.position.y = self.camera.position.y.clamp(
                f64::from(bounds.min_y) + 1.5,
                f64::from(bounds.max_y) + 256.0,
            );
        }
    }

    fn place(&mut self) {
        let Some(hit) = self.target else {
            return;
        };
        let pos = hit.block.offset(hit.face);
        // Don't wall the camera in.
        if pos == BlockPos(self.camera.position.floor().as_ivec3()) {
            return;
        }
        self.client.place_block(pos, self.hotbar[self.selected].1);
    }

    /// How far the world is loaded and drawn, in chunks.
    pub fn set_view_distance(&mut self, chunks: u8) {
        self.view_distance = (i32::from(chunks) * CHUNK_SIZE) as f32;
        self.client.set_view_distance(chunks);
    }

    /// Vertical field of view in degrees.
    pub fn set_fov(&mut self, degrees: f32) {
        self.camera.fov_y = degrees.to_radians();
    }

    pub fn scene(&self) -> Scene {
        Scene {
            camera: self.camera,
            target: self.target.map(|hit| hit.block),
            view_distance: self.view_distance,
            bounds: self.client.bounds(),
            time_of_day: self.time_of_day(),
            eye_light: self.eye_light(),
        }
    }

    /// Whether the world around the player has stopped loading: nothing
    /// arrived or got meshed for a second.
    pub fn is_loaded(&self) -> bool {
        self.joined && self.mesher.backlog() == 0 && self.last_change.elapsed() > LOAD_QUIET
    }

    /// Fraction of the day gone: 0 sunrise, 0.25 noon, 0.5 sunset, 0.75
    /// midnight.
    fn time_of_day(&self) -> f32 {
        let time = self.client.time().unwrap_or(0.0);
        (time.rem_euclid(DAY_LENGTH as f64) / DAY_LENGTH as f64) as f32
    }

    /// The light where the camera is: open sky outside the loaded world.
    fn eye_light(&self) -> Light {
        let pos = BlockPos(self.camera.position.floor().as_ivec3());
        self.client
            .world()
            .chunk(pos.chunk())
            .map_or(Light::SKY, |chunk| chunk.light().get(pos.local()))
    }

    /// Debug information for the window title.
    pub fn status(&self) -> String {
        let p = self.camera.position;
        let block = self.hotbar.get(self.selected).map_or("", |(name, _)| name);
        // Sunrise is 6 o'clock.
        let minutes = ((self.time_of_day() * 24.0 + 6.0) * 60.0) as u32 % (24 * 60);
        format!(
            "{:.0} {:.0} {:.0} · {:02}:{:02} · {block} · {} chunks, {} meshing",
            p.x,
            p.y,
            p.z,
            minutes / 60,
            minutes % 60,
            self.client.world().chunk_count(),
            self.mesher.backlog()
        )
    }

    /// Disconnects and waits for the integrated server to stop.
    pub fn shutdown(self) {
        let Self { client, server, .. } = self;
        drop(client);
        if let Some(server) = server
            && server.join().is_err()
        {
            warn!("the server thread panicked");
        }
    }
}
