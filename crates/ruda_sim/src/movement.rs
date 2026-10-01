//! How players move: walking, jumping, sneaking and flying, held back by
//! solid blocks.
//!
//! Sizes and speeds follow Minecraft's, so players' habits and buildings
//! work the same: a player fits through a gap one block wide and two tall,
//! jumps onto one block but not two, and walks at about 4.3 blocks a second.
//! Speeds are in blocks per tick.

use glam::{DVec3, Vec2};
use ruda_core::{BlockPos, BlockRegistry, WorldBounds};
use ruda_world::World;

/// Width and depth of the player's box, in blocks.
pub const PLAYER_WIDTH: f64 = 0.6;
pub const PLAYER_HEIGHT: f64 = 1.8;
/// How far above the feet the eyes are.
pub const EYE_HEIGHT: f64 = 1.62;
/// How far above the top of the world players can fly, in blocks.
pub const FLY_CEILING: i32 = 256;

/// Speed gained a tick walking on the ground with the key held.
const WALK: f64 = 0.1;
/// Speed gained a tick in the air, where there's little to push against.
const AIR_WALK: f64 = 0.02;
const SPRINT_FACTOR: f64 = 1.3;
const SNEAK_FACTOR: f64 = 0.3;
/// Input counts for a little less than full: with it, walking comes to 4.3
/// blocks a second as in Minecraft.
const INPUT_SCALE: f64 = 0.98;
/// Share of the sideways speed kept from one tick to the next.
const GROUND_KEPT: f64 = 0.546;
const AIR_KEPT: f64 = 0.91;
const GRAVITY: f64 = 0.08;
/// Share of the vertical speed kept: air holds a falling player back, up to
/// 3.92 blocks a tick, 78 a second.
const FALL_KEPT: f64 = 0.98;
/// Up to 1.25 blocks high.
const JUMP_SPEED: f64 = 0.42;
/// Extra forward speed of a jump while sprinting.
const SPRINT_JUMP: f64 = 0.2;
/// Flying comes to 12 blocks a second, 40 when sprinting, up and down too.
const FLY: f64 = 0.24;
const FLY_SPRINT: f64 = 0.8;
const FLY_KEPT: f64 = 0.6;
/// Slower speeds stop, so a player at rest really rests.
const MIN_SPEED: f64 = 0.003;
/// Two presses of jump this many ticks apart or closer switch flying.
const DOUBLE_TAP_TICKS: u8 = 7;
/// A sneaking player doesn't step off an edge higher than this.
const SNEAK_DROP: f64 = 0.6;
/// Sneaking at an edge, movement is cut back in steps of this much until
/// the player stays on it.
const SNEAK_BACK_OFF: f64 = 0.05;
/// How far ahead auto-jump looks for a step.
const AUTO_JUMP_REACH: f64 = 0.4;
/// Allows for rounding errors: boxes this close count as touching, not as
/// overlapping.
const EPSILON: f64 = 1e-7;

/// What a player wants to do in one tick: the keys or sticks they hold and
/// where they look.
#[derive(Clone, Copy, Debug, Default, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayerInput {
    /// Numbers a player's inputs, one a tick, counting up from 1.
    pub seq: u32,
    /// Where the player wants to go, each from −1 to 1: x to the right, y
    /// forward. A stick tilted halfway walks at half the speed.
    pub walk: Vec2,
    /// Turn around the vertical axis in radians, as the camera's: 0 looks
    /// towards −Z and positive values turn right.
    pub yaw: f32,
    /// Look up (positive) or down, in radians. Moving doesn't depend on it.
    pub pitch: f32,
    /// Jump is held, or was pressed since the last tick.
    pub jump: bool,
    /// Jump was pressed since the last tick: two presses close together
    /// switch flying.
    pub jump_pressed: bool,
    pub sneak: bool,
    pub sprint: bool,
}

impl PlayerInput {
    /// The same input with anything out of range made harmless: the server
    /// gets it from clients it can't trust.
    fn sanitized(&self) -> Self {
        let walk = if self.walk.is_finite() {
            // Each way first: the length of huge values overflows.
            self.walk
                .clamp(Vec2::NEG_ONE, Vec2::ONE)
                .clamp_length_max(1.0)
        } else {
            Vec2::ZERO
        };
        let yaw = if self.yaw.is_finite() { self.yaw } else { 0.0 };
        Self { walk, yaw, ..*self }
    }

    /// Horizontal direction to walk in, up to a block long.
    fn direction(&self) -> DVec3 {
        let (forward, right) = (forward(self.yaw), right(self.yaw));
        forward * f64::from(self.walk.y) + right * f64::from(self.walk.x)
    }

    /// Whether the player walks forward fast.
    fn sprints(&self) -> bool {
        self.sprint && self.walk.y > 0.0 && !self.sneak
    }
}

/// Where a player looking at `yaw` faces, horizontally.
fn forward(yaw: f32) -> DVec3 {
    let (sin, cos) = f64::from(yaw).sin_cos();
    DVec3::new(sin, 0.0, -cos)
}

/// The player's right when looking at `yaw`.
fn right(yaw: f32) -> DVec3 {
    let (sin, cos) = f64::from(yaw).sin_cos();
    DVec3::new(cos, 0.0, sin)
}

/// Where a player is and how it moves.
#[derive(Clone, Copy, Debug, PartialEq)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlayerState {
    /// The middle of the bottom of the player's box: where its feet are.
    pub position: DVec3,
    /// In blocks per tick.
    pub velocity: DVec3,
    pub on_ground: bool,
    pub flying: bool,
    /// Ticks left in which another press of jump switches flying.
    pub double_tap: u8,
}

impl PlayerState {
    /// A player standing, or falling, with its feet at `position`.
    pub fn new(position: DVec3) -> Self {
        Self {
            position,
            velocity: DVec3::ZERO,
            on_ground: false,
            flying: false,
            double_tap: 0,
        }
    }

    /// A player hovering with its feet at `position`.
    pub fn flying(position: DVec3) -> Self {
        Self {
            flying: true,
            ..Self::new(position)
        }
    }

    pub fn eye(&self) -> DVec3 {
        self.position + DVec3::Y * EYE_HEIGHT
    }

    /// Whether a solid block at `pos` would be inside the player.
    pub fn overlaps(&self, pos: BlockPos) -> bool {
        Aabb::player(self.position).intersects(Aabb::block(pos))
    }

    /// Moves the player one tick on. `solid` tells whether the block at a
    /// position stops players, or `None` if it isn't loaded: the server and
    /// the client both use [`solid_in`].
    pub fn step(&mut self, input: &PlayerInput, solid: impl Fn(BlockPos) -> Option<bool>) {
        let input = input.sanitized();
        // Until the world around is there, the player waits in place rather
        // than falling into what isn't loaded.
        if !self.is_loaded(&solid) {
            self.velocity = DVec3::ZERO;
            return;
        }

        self.double_tap = self.double_tap.saturating_sub(1);
        if input.jump_pressed {
            if self.double_tap > 0 {
                self.flying = !self.flying;
                self.double_tap = 0;
            } else {
                self.double_tap = DOUBLE_TAP_TICKS;
            }
        }
        self.velocity = self
            .velocity
            .map(|speed| if speed.abs() < MIN_SPEED { 0.0 } else { speed });

        let flying = self.flying;
        let on_ground = self.on_ground;
        if flying {
            let speed = if input.sprint { FLY_SPRINT } else { FLY };
            let vertical = f64::from(u8::from(input.jump)) - f64::from(u8::from(input.sneak));
            self.velocity += (input.direction() + DVec3::Y * vertical) * speed;
        } else {
            if input.jump && on_ground {
                self.velocity.y = JUMP_SPEED;
                if input.sprints() {
                    self.velocity += forward(input.yaw) * SPRINT_JUMP;
                }
            }
            let mut speed = if on_ground { WALK } else { AIR_WALK };
            if input.sprints() {
                speed *= SPRINT_FACTOR;
            }
            if input.sneak {
                speed *= SNEAK_FACTOR;
            }
            self.velocity += input.direction() * INPUT_SCALE * speed;
        }

        let body = Aabb::player(self.position);
        let mut wanted = self.velocity;
        if input.sneak && on_ground && !flying && wanted.y <= 0.0 {
            wanted = back_off_from_edge(body, wanted, &solid);
        }
        let moved = collide(body, wanted, &solid);
        self.position += moved;
        // Running into something stops the movement that way.
        for axis in 0..3 {
            if moved[axis] != wanted[axis] {
                self.velocity[axis] = 0.0;
            }
        }
        self.on_ground = wanted.y < 0.0 && moved.y != wanted.y;

        if flying {
            self.velocity *= FLY_KEPT;
            if self.on_ground {
                self.flying = false;
            }
        } else {
            self.velocity.y = (self.velocity.y - GRAVITY) * FALL_KEPT;
            let kept = if on_ground { GROUND_KEPT } else { AIR_KEPT };
            self.velocity.x *= kept;
            self.velocity.z *= kept;
        }
    }

    /// Whether a jump would take the player up onto the block it is walking
    /// into. Clients that jump for their players press jump then.
    pub fn should_auto_jump(
        &self,
        input: &PlayerInput,
        solid: impl Fn(BlockPos) -> Option<bool>,
    ) -> bool {
        let input = input.sanitized();
        let direction = input.direction();
        if self.flying || !self.on_ground || input.sneak || direction.length_squared() < 0.01 {
            return false;
        }
        let body = Aabb::player(self.position);
        let ahead = body.translate(direction.normalize() * AUTO_JUMP_REACH);
        let up = DVec3::Y;
        !is_free(ahead, &solid)
            && is_free(ahead.translate(up), &solid)
            && is_free(body.translate(up), &solid)
    }

    /// Whether the blocks around the player, and under it, are loaded.
    fn is_loaded(&self, solid: &impl Fn(BlockPos) -> Option<bool>) -> bool {
        let body = Aabb::player(self.position);
        let around = Aabb {
            min: body.min - DVec3::ONE,
            max: body.max + DVec3::new(1.0, 0.0, 1.0),
        };
        around.cells().all(|pos| solid(pos).is_some())
    }
}

/// Whether the block at `pos` stops players, as both the client and the
/// server see it: `None` while its chunk isn't loaded. Above the world is
/// open air up to [`FLY_CEILING`]; below it, where there is nothing to
/// stand on, a floor.
pub fn solid_in<'a>(
    world: &'a World,
    blocks: &'a BlockRegistry,
    bounds: WorldBounds,
) -> impl Fn(BlockPos) -> Option<bool> + 'a {
    move |pos| {
        if pos.0.y < bounds.min_y || pos.0.y > bounds.max_y + FLY_CEILING {
            Some(true)
        } else if pos.0.y > bounds.max_y {
            Some(false)
        } else {
            world.block(pos).map(|block| blocks.is_solid(block))
        }
    }
}

/// Cuts back a sneaking player's movement so it doesn't step off an edge.
fn back_off_from_edge(
    body: Aabb,
    wanted: DVec3,
    solid: &impl Fn(BlockPos) -> Option<bool>,
) -> DVec3 {
    let unsupported =
        |x: f64, z: f64| is_free(body.translate(DVec3::new(x, -SNEAK_DROP, z)), solid);
    let back_off = |speed: f64| {
        if speed.abs() <= SNEAK_BACK_OFF {
            0.0
        } else {
            speed - SNEAK_BACK_OFF.copysign(speed)
        }
    };
    let (mut x, mut z) = (wanted.x, wanted.z);
    while x != 0.0 && unsupported(x, 0.0) {
        x = back_off(x);
    }
    while z != 0.0 && unsupported(0.0, z) {
        z = back_off(z);
    }
    while x != 0.0 && z != 0.0 && unsupported(x, z) {
        x = back_off(x);
        z = back_off(z);
    }
    DVec3::new(x, wanted.y, z)
}

/// How far `body` gets towards `wanted` before solid blocks stop it: up or
/// down first, then sideways, the larger way first. Blocks it already
/// overlaps don't stop it, so a player can get out of a block put into it.
fn collide(body: Aabb, wanted: DVec3, solid: &impl Fn(BlockPos) -> Option<bool>) -> DVec3 {
    let obstacles: Vec<Aabb> = body
        .expand_towards(wanted)
        .cells()
        .filter(|&pos| solid(pos) != Some(false))
        .map(Aabb::block)
        .collect();
    let order = if wanted.x.abs() < wanted.z.abs() {
        [1, 2, 0]
    } else {
        [1, 0, 2]
    };
    let mut body = body;
    let mut moved = DVec3::ZERO;
    for axis in order {
        let distance = obstacles.iter().fold(wanted[axis], |distance, obstacle| {
            body.clip(obstacle, axis, distance)
        });
        moved[axis] = distance;
        let mut offset = DVec3::ZERO;
        offset[axis] = distance;
        body = body.translate(offset);
    }
    moved
}

/// Whether nothing solid, or not loaded, overlaps `area`.
fn is_free(area: Aabb, solid: &impl Fn(BlockPos) -> Option<bool>) -> bool {
    !area
        .cells()
        .any(|pos| solid(pos) != Some(false) && Aabb::block(pos).intersects(area))
}

/// A box with edges along the axes.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Aabb {
    min: DVec3,
    max: DVec3,
}

impl Aabb {
    fn player(feet: DVec3) -> Self {
        let half = PLAYER_WIDTH / 2.0;
        Self {
            min: feet - DVec3::new(half, 0.0, half),
            max: feet + DVec3::new(half, PLAYER_HEIGHT, half),
        }
    }

    fn block(pos: BlockPos) -> Self {
        let min = pos.0.as_dvec3();
        Self {
            min,
            max: min + DVec3::ONE,
        }
    }

    fn translate(self, by: DVec3) -> Self {
        Self {
            min: self.min + by,
            max: self.max + by,
        }
    }

    /// Grown to cover everything it passes moving by `by`.
    fn expand_towards(self, by: DVec3) -> Self {
        Self {
            min: self.min + by.min(DVec3::ZERO),
            max: self.max + by.max(DVec3::ZERO),
        }
    }

    /// Whether the two overlap along `axis` by more than a rounding error.
    fn overlaps_along(&self, other: &Aabb, axis: usize) -> bool {
        self.min[axis] < other.max[axis] - EPSILON && self.max[axis] > other.min[axis] + EPSILON
    }

    fn intersects(&self, other: Aabb) -> bool {
        (0..3).all(|axis| self.overlaps_along(&other, axis))
    }

    /// How far this box can move along `axis`, up to `distance`, before it
    /// runs into `obstacle`.
    fn clip(&self, obstacle: &Aabb, axis: usize, distance: f64) -> f64 {
        let (a, b) = ((axis + 1) % 3, (axis + 2) % 3);
        if !self.overlaps_along(obstacle, a) || !self.overlaps_along(obstacle, b) {
            return distance;
        }
        if distance > 0.0 && obstacle.min[axis] >= self.max[axis] - EPSILON {
            distance.min(obstacle.min[axis] - self.max[axis])
        } else if distance < 0.0 && obstacle.max[axis] <= self.min[axis] + EPSILON {
            distance.max(obstacle.max[axis] - self.min[axis])
        } else {
            distance
        }
    }

    /// The blocks it reaches into.
    fn cells(self) -> impl Iterator<Item = BlockPos> {
        let min = self.min.floor().as_ivec3();
        let max = self.max.floor().as_ivec3();
        (min.y..=max.y).flat_map(move |y| {
            (min.z..=max.z).flat_map(move |z| (min.x..=max.x).map(move |x| BlockPos::new(x, y, z)))
        })
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use super::*;

    /// Ground with its top at y = 0, extra blocks on it, and nothing
    /// loaded beyond 100 blocks from the middle.
    #[derive(Default)]
    struct Terrain {
        blocks: HashSet<BlockPos>,
        holes: HashSet<BlockPos>,
    }

    impl Terrain {
        fn with(mut self, x: i32, y: i32, z: i32) -> Self {
            self.blocks.insert(BlockPos::new(x, y, z));
            self
        }

        fn without(mut self, x: i32, y: i32, z: i32) -> Self {
            self.holes.insert(BlockPos::new(x, y, z));
            self
        }

        fn solid(&self) -> impl Fn(BlockPos) -> Option<bool> + '_ {
            |pos| {
                if pos.0.x.abs() > 100 || pos.0.z.abs() > 100 {
                    return None;
                }
                let ground = pos.0.y < 0 && !self.holes.contains(&pos);
                Some(ground || self.blocks.contains(&pos))
            }
        }
    }

    fn input(walk: Vec2) -> PlayerInput {
        PlayerInput {
            walk,
            ..Default::default()
        }
    }

    const FORWARD: Vec2 = Vec2::Y;

    fn run(state: &mut PlayerState, input: &PlayerInput, terrain: &Terrain, ticks: usize) {
        for _ in 0..ticks {
            state.step(input, terrain.solid());
        }
    }

    /// Blocks covered in the tick after `ticks` of `input`.
    fn speed(input: &PlayerInput, start: PlayerState, ticks: usize) -> f64 {
        let terrain = Terrain::default();
        let mut state = start;
        run(&mut state, input, &terrain, ticks);
        let before = state.position;
        state.step(input, terrain.solid());
        (state.position - before).length()
    }

    #[test]
    fn falls_and_lands_on_the_ground() {
        let terrain = Terrain::default();
        let mut state = PlayerState::new(DVec3::new(0.5, 20.0, 0.5));
        run(&mut state, &PlayerInput::default(), &terrain, 100);
        assert!(state.on_ground);
        assert!(state.position.y.abs() < 1e-9, "{}", state.position.y);
        run(&mut state, &PlayerInput::default(), &terrain, 20);
        assert!(state.on_ground && state.position.y.abs() < 1e-9);
    }

    #[test]
    fn falls_no_faster_than_the_air_allows() {
        let mut state = PlayerState::new(DVec3::new(0.5, 3000.0, 0.5));
        run(
            &mut state,
            &PlayerInput::default(),
            &Terrain::default(),
            400,
        );
        assert!(
            (state.velocity.y + 3.92).abs() < 0.01,
            "{}",
            state.velocity.y
        );
    }

    #[test]
    fn walks_sprints_and_sneaks_as_fast_as_in_minecraft() {
        let ground = PlayerState::new(DVec3::new(-50.0, 0.0, 0.5));
        let per_second = |input: PlayerInput| speed(&input, ground, 60) * 20.0;
        let walk = per_second(input(FORWARD));
        assert!((walk - 4.317).abs() < 0.01, "{walk}");
        let sprint = per_second(PlayerInput {
            sprint: true,
            ..input(FORWARD)
        });
        assert!((sprint - 5.612).abs() < 0.01, "{sprint}");
        let sneak = per_second(PlayerInput {
            sneak: true,
            ..input(FORWARD)
        });
        assert!((sneak - 1.295).abs() < 0.01, "{sneak}");
        // Sprinting only works forward.
        let back = per_second(PlayerInput {
            sprint: true,
            ..input(-FORWARD)
        });
        assert!((back - walk).abs() < 1e-9);
    }

    #[test]
    fn faces_where_the_camera_looks() {
        let terrain = Terrain::default();
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &input(FORWARD), &terrain, 10);
        // Yaw 0 looks towards −Z.
        assert!(state.position.z < 0.0 && (state.position.x - 0.5).abs() < 1e-9);

        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        let right = PlayerInput {
            yaw: std::f32::consts::FRAC_PI_2,
            ..input(FORWARD)
        };
        run(&mut state, &right, &terrain, 10);
        assert!(state.position.x > 1.0 && (state.position.z - 0.5).abs() < 1e-6);
    }

    #[test]
    fn walls_stop_the_player() {
        // A wall two blocks tall across the way at z = −3.
        let mut terrain = Terrain::default();
        for x in -3..3 {
            terrain = terrain.with(x, 0, -3).with(x, 1, -3);
        }
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &input(FORWARD), &terrain, 40);
        assert!((state.position.z - (-2.0 + PLAYER_WIDTH / 2.0)).abs() < 1e-6);
        assert_eq!(state.velocity.z, 0.0);

        // Jumping doesn't get over two blocks.
        let jump = PlayerInput {
            jump: true,
            ..input(FORWARD)
        };
        run(&mut state, &jump, &terrain, 40);
        assert!(state.position.z > -2.0 && state.position.y < 1.5);
    }

    #[test]
    fn jumps_a_block_and_a_quarter_high() {
        let terrain = Terrain::default();
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &PlayerInput::default(), &terrain, 2);
        let jump = PlayerInput {
            jump: true,
            ..Default::default()
        };
        let mut highest: f64 = 0.0;
        for _ in 0..20 {
            state.step(&jump, terrain.solid());
            highest = highest.max(state.position.y);
        }
        assert!((highest - 1.25).abs() < 0.01, "{highest}");
    }

    #[test]
    fn jumps_onto_a_block() {
        // A platform a block high from z = −2 on.
        let mut terrain = Terrain::default();
        for x in -3..3 {
            for z in -20..-2 {
                terrain = terrain.with(x, 0, z);
            }
        }
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        let jump = PlayerInput {
            jump: true,
            ..input(FORWARD)
        };
        run(&mut state, &jump, &terrain, 20);
        run(&mut state, &input(FORWARD), &terrain, 20);
        assert!(state.position.z < -2.5);
        assert!((state.position.y - 1.0).abs() < 1e-9 && state.on_ground);
    }

    #[test]
    fn fits_through_a_gap_one_block_wide_and_two_tall() {
        // A corridor along −Z: walls at x = −1 and x = 1, a roof at y = 2.
        let mut terrain = Terrain::default();
        for z in -10..-2 {
            terrain = terrain
                .with(-1, 0, z)
                .with(-1, 1, z)
                .with(1, 0, z)
                .with(1, 1, z)
                .with(0, 2, z);
        }
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &input(FORWARD), &terrain, 80);
        assert!(state.position.z < -10.0, "{}", state.position.z);
    }

    #[test]
    fn sneaking_keeps_the_player_on_the_edge() {
        // A pit from z = −2 on.
        let mut terrain = Terrain::default();
        for x in -3..3 {
            for z in -8..-1 {
                for y in -3..0 {
                    terrain = terrain.without(x, y, z);
                }
            }
        }
        let sneak = PlayerInput {
            sneak: true,
            ..input(FORWARD)
        };
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &sneak, &terrain, 100);
        assert!(state.on_ground && state.position.y == 0.0);
        // It may lean out, but no farther than its box still touches the edge.
        assert!(state.position.z > -1.0 - PLAYER_WIDTH / 2.0);
        assert!(state.position.z < -0.9);

        run(&mut state, &input(FORWARD), &terrain, 20);
        assert!(state.position.y < -1.0);
    }

    #[test]
    fn double_jump_switches_flying() {
        let terrain = Terrain::default();
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        let press = PlayerInput {
            jump: true,
            jump_pressed: true,
            ..Default::default()
        };
        let release = PlayerInput::default();
        run(&mut state, &release, &terrain, 2);
        state.step(&press, terrain.solid());
        run(&mut state, &release, &terrain, 3);
        state.step(&press, terrain.solid());
        assert!(state.flying);
        run(&mut state, &release, &terrain, 40);
        // It hovers.
        assert!(state.flying && state.position.y > 0.5);

        // Two presses far apart don't count.
        state.step(&press, terrain.solid());
        run(&mut state, &release, &terrain, 10);
        state.step(&press, terrain.solid());
        assert!(state.flying);

        // Coming down to the ground ends the flight.
        let down = PlayerInput {
            sneak: true,
            ..Default::default()
        };
        run(&mut state, &down, &terrain, 40);
        assert!(!state.flying && state.on_ground);
    }

    #[test]
    fn flies_as_fast_as_free_flight_did() {
        let hovering = PlayerState::flying(DVec3::new(-50.0, 10.0, 0.5));
        let per_second = |input: PlayerInput| speed(&input, hovering, 40) * 20.0;
        assert!((per_second(input(FORWARD)) - 12.0).abs() < 0.01);
        let sprint = PlayerInput {
            sprint: true,
            ..input(FORWARD)
        };
        assert!((per_second(sprint) - 40.0).abs() < 0.01);
        let up = PlayerInput {
            jump: true,
            ..Default::default()
        };
        assert!((per_second(up) - 12.0).abs() < 0.01);
    }

    #[test]
    fn waits_where_the_world_isnt_loaded() {
        let terrain = Terrain::default();
        let mut state = PlayerState::new(DVec3::new(150.5, 20.0, 0.5));
        run(&mut state, &input(FORWARD), &terrain, 20);
        assert_eq!(state.position, DVec3::new(150.5, 20.0, 0.5));

        // Nor does it walk into it.
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, -98.5));
        run(&mut state, &input(FORWARD), &terrain, 100);
        assert!(state.position.z > -100.0);
    }

    #[test]
    fn gets_out_of_a_block_put_into_it() {
        let terrain = Terrain::default().with(0, 0, 0).with(0, 1, 0);
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        assert!(state.overlaps(BlockPos::new(0, 1, 0)));
        assert!(!state.overlaps(BlockPos::new(1, 0, 0)));
        assert!(!state.overlaps(BlockPos::new(0, -1, 0)));
        run(&mut state, &input(FORWARD), &terrain, 20);
        assert!(state.position.z < -1.0 && state.position.y == 0.0);
    }

    #[test]
    fn auto_jumps_onto_steps_only() {
        let terrain = Terrain::default().with(0, 0, -1);
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &PlayerInput::default(), &terrain, 2);
        let towards = input(FORWARD);
        assert!(state.should_auto_jump(&towards, terrain.solid()));
        assert!(!state.should_auto_jump(&input(-FORWARD), terrain.solid()));
        assert!(!state.should_auto_jump(&PlayerInput::default(), terrain.solid()));

        let wall = Terrain::default().with(0, 0, -1).with(0, 1, -1);
        assert!(!state.should_auto_jump(&towards, wall.solid()));
        let low_roof = Terrain::default().with(0, 0, -1).with(0, 2, 0);
        assert!(!state.should_auto_jump(&towards, low_roof.solid()));
    }

    #[test]
    fn ignores_nonsense_input() {
        let terrain = Terrain::default();
        let mut state = PlayerState::new(DVec3::new(0.5, 0.0, 0.5));
        run(&mut state, &PlayerInput::default(), &terrain, 2);
        let nonsense = PlayerInput {
            walk: Vec2::new(f32::NAN, 1e30),
            yaw: f32::INFINITY,
            ..Default::default()
        };
        run(&mut state, &nonsense, &terrain, 5);
        assert!(state.position.is_finite());
        // Holding the stick past its end is no faster than all the way.
        let far = PlayerInput {
            walk: Vec2::new(0.0, 1e30),
            ..Default::default()
        };
        let start = PlayerState::new(DVec3::new(-50.0, 0.0, 0.5));
        assert!((speed(&far, start, 60) - speed(&input(FORWARD), start, 60)).abs() < 1e-9);
    }
}
