//! The game itself: an integrated server, the client talking to it, and the
//! player's camera and controls.

use std::collections::VecDeque;
use std::sync::Arc;
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context as _, Result, anyhow};
use glam::{DVec3, IVec3};
use ruda_client::{Client, Event};
use ruda_core::{
    Appearance, BlockId, BlockPos, CHUNK_SIZE, ChunkPos, Content, ContentBuilder, Light,
    ResourceId, WindMap, WorldBounds,
};
use ruda_input::{Action, Input};
use ruda_protocol::{DAY_LENGTH, PlayerInput, REACH, TICK_RATE};
use ruda_render::{
    BlockFaces, Camera, ChunkMesher, CloudSky, Renderer, Scene, cloud_obstacles,
    far_cloud_obstacles, mesh_lod,
};
use ruda_server::ServerConfig;
use ruda_sim::EYE_HEIGHT;
use ruda_world::lod::LodTilePos;
use ruda_world::{RayHit, raycast};
use tracing::{info, warn};

/// Seconds a tick lasts.
const TICK: f64 = 1.0 / TICK_RATE as f64;
/// Most ticks run in one update: after a long hitch the game drops time
/// rather than racing to catch up.
const MAX_TICKS_PER_UPDATE: u32 = 5;
/// Radians of camera turn per unit of mouse movement.
const MOUSE_SENSITIVITY: f32 = 0.0025;
/// How much of the sky clouds cover, and how dense they are, until weather
/// decides it.
const CLOUD_COVER: f32 = 0.35;
const CLOUD_DENSITY: f32 = 1.0;
/// Far-away tiles turned into geometry per frame.
const LOD_TILES_PER_FRAME: usize = 2;
/// See [`Game::is_loaded`].
const LOAD_QUIET: Duration = Duration::from_secs(1);
/// The world is shown once the chunks this many chunks around the player are
/// there: until then, the far-away look of the world stands in for them,
/// coarse and stretched. See [`Game::is_ready`].
const READY_RADIUS: i32 = 2;
/// Or after this long, whatever is missing.
const READY_AT_MOST: Duration = Duration::from_secs(10);
/// Cells in the hotbar, one for each number key.
pub const HOTBAR_SLOTS: usize = 10;
/// The farthest the integrated server streams the world, in chunks.
pub const MAX_VIEW_DISTANCE: u8 = 32;

/// The world behind the title menu: the one the title picture was taken in,
/// from the same spot, at sunset.
const PANORAMA_SEED: u64 = 7;
const PANORAMA_TIME: u64 = 11_500;
const PANORAMA_CAMERA: CameraStart = CameraStart {
    position: DVec3::new(960.0, 128.0, -350.0),
    yaw: 256.0,
    pitch: -6.0,
};
/// While the panorama loads out of sight, how long a frame may spend turning
/// far-away tiles into geometry.
const HIDDEN_LOD_BUDGET: Duration = Duration::from_millis(12);
/// Shown after this long whatever is still missing.
const PANORAMA_AT_MOST: Duration = Duration::from_secs(20);
/// It is only a backdrop, so it is not worth drawing far.
const PANORAMA_VIEW_DISTANCE: u8 = 6;
/// The camera turns this far to either side of where it started, in
/// degrees, and is back after this many seconds.
const PANORAMA_PAN: f32 = 35.0;
const PANORAMA_PAN_PERIOD: f32 = 240.0;
/// A slow nod up and down, in degrees and seconds.
const PANORAMA_NOD: f32 = 1.5;
const PANORAMA_NOD_PERIOD: f32 = 37.0;
/// The sun creeps back and forth around the horizon instead of setting: a
/// fraction of the day, and seconds.
const PANORAMA_SUN: f32 = 0.015;
const PANORAMA_SUN_PERIOD: f32 = 360.0;

#[derive(Clone, Copy, Debug)]
pub struct GameConfig {
    pub seed: u64,
    /// In chunks.
    pub view_distance: u8,
    /// Vertical field of view in degrees.
    pub fov: f32,
    /// Where the camera starts, the player flying, instead of the spawn
    /// point.
    pub camera: Option<CameraStart>,
    /// The time the world starts at, in ticks; see [`DAY_LENGTH`].
    pub time: Option<u64>,
    /// How far the far-away look of the world reaches, in blocks; 0 for
    /// none.
    pub lod_distance: u16,
    pub clouds: bool,
    /// Jump onto blocks the player walks into.
    pub auto_jump: bool,
    /// How fast the mouse turns the camera, in percent of the normal speed.
    pub mouse_sensitivity: u8,
    /// The view sways with the player's steps.
    pub view_bobbing: bool,
    /// Nobody plays: the camera drifts over a world picked for its looks, as
    /// a backdrop for the title menu.
    pub panorama: bool,
}

impl GameConfig {
    /// The same settings, but for the title menu's backdrop.
    pub fn for_panorama(self) -> Self {
        Self {
            seed: PANORAMA_SEED,
            view_distance: self.view_distance.min(PANORAMA_VIEW_DISTANCE),
            camera: Some(PANORAMA_CAMERA),
            time: Some(PANORAMA_TIME),
            auto_jump: false,
            view_bobbing: false,
            panorama: true,
            ..self
        }
    }
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

/// Radians of camera turn per unit of mouse movement at `percent` of the
/// normal speed.
fn sensitivity(percent: u8) -> f32 {
    MOUSE_SENSITIVITY * f32::from(percent) / 100.0
}

/// The blocks of `content` a player can place, each once whatever its states.
pub fn placeable_blocks(content: &Content) -> Vec<BlockId> {
    let blocks = content.blocks();
    blocks
        .iter()
        .filter(|&(id, def)| {
            blocks.state(id) == 0
                && def.breakable
                && !matches!(def.appearance, Appearance::Invisible)
                && def.id.namespace() != ResourceId::ENGINE
        })
        .map(|(id, _)| id)
        .collect()
}

/// The next far-away tile to turn into geometry this frame, if the frame's
/// share is not used up: a couple of tiles, or as many as fit in a moment
/// while nobody sees the world yet (`hidden`, with `since` when it began).
/// Tiles left in the queue wait for the next frame.
fn next_lod_tile(
    waiting: &mut VecDeque<LodTilePos>,
    uploaded: usize,
    hidden: bool,
    since: Instant,
) -> Option<LodTilePos> {
    let used_up = if hidden {
        since.elapsed() > HIDDEN_LOD_BUDGET
    } else {
        uploaded >= LOD_TILES_PER_FRAME
    };
    if used_up { None } else { waiting.pop_front() }
}

/// How the view sways as the player walks, as in Minecraft.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
struct Bob {
    /// Grows as the player walks, by one every other step.
    walked: f64,
    /// How strongly the view sways: up to 0.1 when walking on the ground,
    /// less sneaking, and dying away in the air.
    amount: f64,
}

impl Bob {
    /// After a tick in which the player moved by `moved`.
    fn step(self, moved: DVec3, on_ground: bool) -> Self {
        let distance = moved.x.hypot(moved.z);
        let amount = if on_ground { distance.min(0.1) } else { 0.0 };
        Self {
            walked: self.walked + distance * 0.6,
            amount: self.amount + (amount - self.amount) * 0.4,
        }
    }

    fn lerp(self, to: Self, alpha: f64) -> Self {
        Self {
            walked: self.walked + (to.walked - self.walked) * alpha,
            amount: self.amount + (to.amount - self.amount) * alpha,
        }
    }

    /// Moves and tilts `camera` with the steps: a little sideways and up,
    /// leaning and dipping with them.
    fn sway(self, camera: &mut Camera) {
        let phase = self.walked * std::f64::consts::PI;
        let (sin, cos) = phase.sin_cos();
        let right = camera.right().as_dvec3();
        let up = right.cross(camera.forward().as_dvec3());
        camera.position += right * (-sin * self.amount * 0.5) + up * (cos.abs() * self.amount);
        camera.roll += (sin * self.amount * 3.0).to_radians() as f32;
        camera.pitch -= ((phase - 0.2).cos().abs() * self.amount * 5.0).to_radians() as f32;
    }
}

/// What the window should do after an update.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Control {
    Continue,
    Pause,
    Inventory,
}

#[derive(Debug)]
pub struct Game {
    client: Client,
    server: Option<JoinHandle<()>>,
    mesher: ChunkMesher,
    pub input: Input,
    camera: Camera,
    /// What each cell of the hotbar holds; all start empty.
    hotbar: Vec<Option<BlockId>>,
    selected: usize,
    target: Option<RayHit>,
    view_distance: f32,
    faces: Arc<BlockFaces>,
    /// In blocks.
    lod_distance: f32,
    /// Far-away tiles that arrived and still need geometry, oldest first.
    lod_waiting: VecDeque<LodTilePos>,
    clouds: bool,
    /// Until weather decides it, a steady breeze.
    wind: WindMap,
    joined: bool,
    start: Option<CameraStart>,
    /// When chunks last arrived, left or got new geometry.
    last_change: Instant,
    started: Instant,
    /// Seconds since the last tick.
    since_tick: f64,
    /// Jump was pressed since the last tick.
    jump_pressed: bool,
    auto_jump: bool,
    /// Radians of camera turn per unit of mouse movement.
    sensitivity: f32,
    /// How far through the tick the frame is, from 0 to 1.
    alpha: f64,
    /// The sway of the view a tick ago and now.
    bob: [Bob; 2],
    view_bobbing: bool,
    /// See [`GameConfig::panorama`].
    panorama: bool,
    /// When the panorama's world had loaded and became worth showing; its
    /// motion counts from here, so it starts as the title picture was
    /// taken.
    shown_at: Option<Instant>,
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
            // Mixed, so the seed itself isn't given away.
            sky_seed: config
                .seed
                .wrapping_mul(0x9e37_79b9_7f4a_7c15)
                .rotate_left(29),
            spawn: config
                .camera
                .map(|start| start.position - DVec3::Y * EYE_HEIGHT),
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
        client.set_lod_distance(config.lod_distance);

        let faces = Arc::new(renderer.load_block_textures(&content));
        renderer.set_sky_textures(ruda_base::SUN, ruda_base::MOON);
        let hotbar = vec![None; HOTBAR_SLOTS];
        info!(seed = config.seed, "world started");
        let mut camera = Camera::new(DVec3::new(0.5, 100.0, 0.5));
        camera.fov_y = config.fov.to_radians();
        Ok(Self {
            client,
            server: Some(server),
            mesher: ChunkMesher::new(Arc::clone(&faces)),
            faces,
            lod_distance: f32::from(config.lod_distance),
            lod_waiting: VecDeque::new(),
            clouds: config.clouds,
            wind: WindMap::BREEZE,
            input: Input::default(),
            camera,
            hotbar,
            selected: 0,
            target: None,
            view_distance: (i32::from(config.view_distance) * CHUNK_SIZE) as f32,
            joined: false,
            start: config.camera,
            last_change: Instant::now(),
            started: Instant::now(),
            since_tick: 0.0,
            jump_pressed: false,
            auto_jump: config.auto_jump,
            sensitivity: sensitivity(config.mouse_sensitivity),
            alpha: 0.0,
            bob: [Bob::default(); 2],
            view_bobbing: config.view_bobbing,
            panorama: config.panorama,
            shown_at: None,
        })
    }

    /// Advances the game by `dt` seconds, running the ticks due. Mouse
    /// movement only turns the camera and clicks only act while the cursor
    /// is grabbed.
    pub fn update(
        &mut self,
        dt: f64,
        cursor_grabbed: bool,
        renderer: &mut Renderer,
    ) -> Result<Control> {
        // Chunks whose blocks may reach the clouds differently now.
        let mut reaching = Vec::new();
        for event in self.client.update() {
            if matches!(event, Event::ChunkLoaded(_) | Event::ChunkUnloaded(_)) {
                self.last_change = Instant::now();
            }
            match event {
                Event::Joined => {
                    match self.start {
                        Some(start) => {
                            self.camera.yaw = 0.0;
                            self.camera.pitch = 0.0;
                            self.camera
                                .rotate(start.yaw.to_radians(), start.pitch.to_radians());
                        }
                        None => self.camera.pitch = -0.3,
                    }
                    self.joined = true;
                }
                Event::ChunkLoaded(pos) => {
                    self.mesher.chunk_loaded(pos);
                    reaching.push(pos);
                }
                Event::ChunkUnloaded(pos) => {
                    self.mesher.forget(pos);
                    renderer.remove_chunk(pos);
                    renderer.set_cloud_obstacles(pos, None);
                }
                Event::BlockChanged(pos) => {
                    reaching.push(pos.chunk());
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
                Event::LodLoaded(pos) => {
                    if !self.lod_waiting.contains(&pos) {
                        self.lod_waiting.push_back(pos);
                    }
                }
                Event::LodUnloaded(pos) => {
                    self.lod_waiting.retain(|&waiting| waiting != pos);
                    renderer.remove_lod(pos);
                    renderer.set_far_cloud_obstacles(pos, None);
                }
                Event::Disconnected { reason } => return Err(anyhow!("disconnected: {reason}")),
            }
        }
        for pos in reaching {
            self.update_cloud_obstacles(pos, renderer);
        }

        let look = self.input.take_look();
        if cursor_grabbed {
            self.camera
                .rotate(look.x * self.sensitivity, -look.y * self.sensitivity);
        }
        let pressed = self.input.take_pressed();
        self.jump_pressed |= pressed.contains(&Action::Jump);
        self.since_tick += dt;
        let mut ticks = 0;
        while self.since_tick >= TICK {
            if ticks == MAX_TICKS_PER_UPDATE {
                self.since_tick = 0.0;
                break;
            }
            self.since_tick -= TICK;
            self.tick();
            ticks += 1;
        }
        // Between the last two ticks; turning goes straight to the camera.
        self.alpha = self.since_tick / TICK;
        if let Some(feet) = self.client.player_position(self.alpha) {
            self.camera.position = feet + DVec3::Y * EYE_HEIGHT;
        }
        if self.panorama {
            // Only once everything is there: a world that builds up in front
            // of the player looks worse than waiting a little.
            if self.shown_at.is_none()
                && (self.is_loaded() || self.started.elapsed() > PANORAMA_AT_MOST)
            {
                self.shown_at = Some(Instant::now());
            }
            self.pan();
        }

        // Aim a little short of the reach limit: the server measures to the
        // block's centre, the ray to its nearest face.
        let client = &self.client;
        // Nobody aims in the panorama.
        self.target = (!self.panorama)
            .then(|| {
                raycast(
                    self.camera.position,
                    self.camera.forward().as_dvec3(),
                    REACH - 1.0,
                    |pos| client.is_targetable(pos),
                )
            })
            .flatten();

        let mut control = Control::Continue;
        for action in pressed {
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
                Action::Inventory => control = Control::Inventory,
                _ => {}
            }
        }
        // The wheel walks through the hotbar, away from the player meaning
        // back.
        let steps = self.input.take_scroll();
        if cursor_grabbed && steps != 0 && !self.hotbar.is_empty() {
            let len = self.hotbar.len() as i32;
            self.selected = (self.selected as i32 - steps).rem_euclid(len) as usize;
        }

        let center = BlockPos(self.camera.position.floor().as_ivec3()).chunk();
        self.mesher.schedule(self.client.world(), center);
        for (pos, mesh) in self.mesher.finished() {
            renderer.upload_chunk(pos, &mesh);
            self.last_change = Instant::now();
        }
        // A couple of far-away tiles a frame keeps loading smooth, unless
        // nobody sees the world yet: then as many as fit in a moment.
        let hidden = self.panorama && self.shown_at.is_none();
        let lod_start = Instant::now();
        let mut uploaded = 0;
        while let Some(pos) = next_lod_tile(&mut self.lod_waiting, uploaded, hidden, lod_start) {
            uploaded += 1;
            if let Some(tile) = self.client.lod(pos) {
                renderer.upload_lod(pos, &mesh_lod(tile, &self.faces));
                renderer.set_far_cloud_obstacles(pos, far_cloud_obstacles(tile));
                self.last_change = Instant::now();
            }
        }
        Ok(control)
    }

    /// Moves the player by what the player holds: one tick of the game.
    fn tick(&mut self) {
        let jump_pressed = std::mem::take(&mut self.jump_pressed);
        let mut input = PlayerInput {
            walk: self.input.walk(),
            yaw: self.camera.yaw,
            pitch: self.camera.pitch,
            jump: jump_pressed || self.input.is_held(Action::Jump),
            jump_pressed,
            sneak: self.input.is_held(Action::Sneak),
            sprint: self.input.is_held(Action::Sprint),
            ..Default::default()
        };
        if self.auto_jump && self.client.should_auto_jump(&input) {
            input.jump = true;
        }
        let before = self.client.player().map(|player| player.position);
        self.client.tick_player(input);
        let now = self.bob[1];
        self.bob[0] = now;
        if let (Some(before), Some(player)) = (before, self.client.player()) {
            self.bob[1] = now.step(player.position - before, player.on_ground);
        }
    }

    fn place(&mut self) {
        let Some(hit) = self.target else {
            return;
        };
        let pos = hit.block.offset(hit.face);
        // Nothing is placed with an empty hand.
        let Some(block) = self.hotbar[self.selected] else {
            return;
        };
        // A torch hangs on the wall or stands on the floor it was put against.
        if let Some(block) = self.client.content().blocks().placed(block, hit.face) {
            self.client.place_block(pos, block);
        }
    }

    /// Turns the camera slowly this way and that over the panorama.
    fn pan(&mut self) {
        let Some(start) = self.start.filter(|_| self.joined) else {
            return;
        };
        let phase = |period: f32| self.panorama_seconds() / period * std::f32::consts::TAU;
        let yaw = start.yaw + PANORAMA_PAN * phase(PANORAMA_PAN_PERIOD).sin();
        let pitch = start.pitch + PANORAMA_NOD * phase(PANORAMA_NOD_PERIOD).sin();
        self.camera.yaw = 0.0;
        self.camera.pitch = 0.0;
        self.camera.rotate(yaw.to_radians(), pitch.to_radians());
    }

    /// Seconds the panorama has been on show.
    fn panorama_seconds(&self) -> f32 {
        self.shown_at.map_or(0.0, |at| at.elapsed().as_secs_f32())
    }

    /// Whether this is the title menu's backdrop rather than a game.
    pub fn is_panorama(&self) -> bool {
        self.panorama
    }

    /// Whether the world is worth drawing: a game always is, the title
    /// menu's backdrop once it has loaded.
    pub fn is_on_show(&self) -> bool {
        !self.panorama || self.shown_at.is_some()
    }

    /// How far the world is loaded and drawn, in chunks.
    pub fn set_view_distance(&mut self, chunks: u8) {
        let chunks = if self.panorama {
            chunks.min(PANORAMA_VIEW_DISTANCE)
        } else {
            chunks
        };
        self.view_distance = (i32::from(chunks) * CHUNK_SIZE) as f32;
        self.client.set_view_distance(chunks);
    }

    /// Tells the renderer where a chunk's blocks reach the clouds.
    fn update_cloud_obstacles(&self, pos: ChunkPos, renderer: &mut Renderer) {
        if let Some(chunk) = self.client.world().chunk(pos) {
            let blocks = self.client.content().blocks();
            let tops = cloud_obstacles(pos, chunk, |block| blocks.is_solid(block));
            renderer.set_cloud_obstacles(pos, tops);
        }
    }

    /// How fast the mouse turns the camera, in percent of the normal speed.
    pub fn set_mouse_sensitivity(&mut self, percent: u8) {
        self.sensitivity = sensitivity(percent);
    }

    pub fn set_auto_jump(&mut self, auto_jump: bool) {
        self.auto_jump = auto_jump;
    }

    pub fn set_view_bobbing(&mut self, view_bobbing: bool) {
        self.view_bobbing = view_bobbing;
    }

    pub fn set_clouds(&mut self, clouds: bool) {
        self.clouds = clouds;
    }

    /// How far the far-away look of the world reaches, in blocks; 0 for
    /// none.
    pub fn set_lod_distance(&mut self, blocks: u16) {
        self.lod_distance = f32::from(blocks);
        self.client.set_lod_distance(blocks);
    }

    /// Vertical field of view in degrees.
    pub fn set_fov(&mut self, degrees: f32) {
        self.camera.fov_y = degrees.to_radians();
    }

    pub fn scene(&self) -> Scene {
        let mut camera = self.camera;
        if self.view_bobbing {
            self.bob[0].lerp(self.bob[1], self.alpha).sway(&mut camera);
        }
        Scene {
            camera,
            target: self.target.map(|hit| hit.block),
            crosshair: !self.panorama,
            view_distance: self.view_distance,
            bounds: self.client.bounds(),
            time_of_day: self.time_of_day(),
            eye_light: self.eye_light(),
            lod_distance: self.lod_distance,
            clouds: self.clouds.then(|| CloudSky {
                seed: self.client.sky_seed(),
                cover: CLOUD_COVER,
                density: CLOUD_DENSITY,
                drift: self
                    .wind
                    .drift(self.client.time().unwrap_or(0.0) / f64::from(TICK_RATE)),
            }),
        }
    }

    /// Whether the world around the player is there to be shown: the chunks
    /// near the player have arrived and been meshed.
    pub fn is_ready(&self) -> bool {
        if self.started.elapsed() > READY_AT_MOST {
            return true;
        }
        let Some(bounds) = self.client.bounds().filter(|_| self.joined) else {
            return false;
        };
        // Within what the server sends, which is a circle of columns.
        let radius = READY_RADIUS.min(self.view_distance as i32 / CHUNK_SIZE - 1);
        let center = BlockPos(self.camera.position.floor().as_ivec3()).chunk();
        let world = self.client.world();
        (-radius..=radius).all(|y| {
            (-radius..=radius).all(|z| {
                (-radius..=radius).all(|x| {
                    let pos = ChunkPos(center.0 + IVec3::new(x, y, z));
                    x * x + z * z > radius * radius
                        || !bounds.contains_chunk(pos)
                        || world.chunk(pos).is_some() && !self.mesher.is_pending(pos)
                })
            })
        })
    }

    /// Whether the world around the player has stopped loading: nothing
    /// arrived or got meshed for a second.
    pub fn is_loaded(&self) -> bool {
        self.joined && self.mesher.backlog() == 0 && self.last_change.elapsed() > LOAD_QUIET
    }

    /// Fraction of the day gone: 0 sunrise, 0.25 noon, 0.5 sunset, 0.75
    /// midnight.
    fn time_of_day(&self) -> f32 {
        if self.panorama {
            let phase = self.panorama_seconds() / PANORAMA_SUN_PERIOD * std::f32::consts::TAU;
            return PANORAMA_TIME as f32 / DAY_LENGTH as f32 + PANORAMA_SUN * phase.sin();
        }
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
        if self.panorama {
            return String::new();
        }
        let p = self.camera.position;
        let blocks = self.client.content().blocks();
        let block = self
            .hotbar
            .get(self.selected)
            .copied()
            .flatten()
            .map_or("", |block| blocks.get(block).id.path());
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

    /// What each cell of the hotbar holds.
    pub fn hotbar(&self) -> &[Option<BlockId>] {
        &self.hotbar
    }

    /// The hotbar cell in hand.
    pub fn selected(&self) -> usize {
        self.selected
    }

    /// Puts `block` in the hotbar cell `slot`, or empties it.
    pub fn set_hotbar(&mut self, slot: usize, block: Option<BlockId>) {
        if let Some(cell) = self.hotbar.get_mut(slot) {
            *cell = block;
        }
    }

    /// Swaps two hotbar cells.
    pub fn swap_hotbar(&mut self, a: usize, b: usize) {
        if a < self.hotbar.len() && b < self.hotbar.len() {
            self.hotbar.swap(a, b);
        }
    }

    /// The blocks a player can put in the hotbar: every one of the game's
    /// content they can place, each once whatever its states.
    pub fn placeable_blocks(&self) -> Vec<BlockId> {
        placeable_blocks(self.client.content())
    }

    pub fn content(&self) -> &Arc<Content> {
        self.client.content()
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

#[cfg(test)]
mod tests {
    use super::*;

    /// The sway after `ticks` of moving by `moved` a tick.
    fn walk(moved: DVec3, on_ground: bool, ticks: usize, from: Bob) -> Bob {
        (0..ticks).fold(from, |bob, _| bob.step(moved, on_ground))
    }

    #[test]
    fn a_frame_takes_its_share_of_far_tiles_and_leaves_the_rest() {
        let mut waiting: VecDeque<_> = (0..5).map(|x| LodTilePos::new(x, 0)).collect();
        let now = Instant::now();
        let mut taken = Vec::new();
        while let Some(tile) = next_lod_tile(&mut waiting, taken.len(), false, now) {
            taken.push(tile);
        }
        assert_eq!(taken, [LodTilePos::new(0, 0), LodTilePos::new(1, 0)]);
        // Nothing is lost: the rest is next in line.
        assert_eq!(waiting.len(), 3);
        assert_eq!(waiting.front(), Some(&LodTilePos::new(2, 0)));
    }

    #[test]
    fn out_of_sight_all_that_is_waiting_is_taken_within_the_time() {
        let mut waiting: VecDeque<_> = (0..50).map(|x| LodTilePos::new(x, 0)).collect();
        let since = Instant::now();
        let mut taken = 0;
        while next_lod_tile(&mut waiting, taken, true, since).is_some() {
            taken += 1;
        }
        // Far more than a visible frame's share, and none thrown away.
        assert!(taken > LOD_TILES_PER_FRAME);
        assert_eq!(taken + waiting.len(), 50);
        // Once the time is gone, it takes nothing.
        let late = Instant::now() - HIDDEN_LOD_BUDGET * 2;
        assert_eq!(next_lod_tile(&mut waiting, 0, true, late), None);
        assert_eq!(taken + waiting.len(), 50);
    }

    #[test]
    fn the_view_sways_only_while_walking_on_the_ground() {
        let standing = walk(DVec3::ZERO, true, 20, Bob::default());
        let mut camera = Camera::new(DVec3::new(0.5, 70.0, 0.5));
        let still = camera;
        standing.sway(&mut camera);
        assert_eq!(camera.position, still.position);
        assert_eq!((camera.roll, camera.pitch), (still.roll, still.pitch));

        // At a walk, about 0.22 blocks a tick.
        let walking = walk(DVec3::new(0.0, 0.0, -0.22), true, 20, Bob::default());
        assert!((walking.amount - 0.1).abs() < 1e-3);
        assert!(walking.walked > 2.0);
        for alpha in [0.0, 0.3, 0.7] {
            let mut swayed = still;
            Bob::default().lerp(walking, alpha).sway(&mut swayed);
            assert!(swayed.position.distance(still.position) <= 0.12);
            assert!(swayed.roll.abs() <= 0.3_f32.to_radians() + 1e-6);
        }

        // In the air it dies away.
        let jumping = walk(DVec3::new(0.0, 0.0, -0.22), false, 20, walking);
        assert!(jumping.amount < 1e-4);
    }
}
