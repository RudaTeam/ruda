//! How light spreads through the world.
//!
//! Sky light falls straight down without fading until something opaque
//! stops it, and spreads sideways and upwards losing a level per block.
//! Light from blocks spreads in every direction losing a level per block,
//! each colour on its own, so a red and a blue lamp together make purple.
//!
//! Chunks get lit from the top of the world down: a chunk's sky light comes
//! from the bottom of the chunk above it. Only lit chunks take part; light
//! reaches a chunk lit later when that chunk pulls it in from its neighbours.

use std::collections::{HashSet, VecDeque};
use std::ops::RangeInclusive;

use glam::IVec3;
use ruda_core::{
    BlockId, BlockPos, BlockRegistry, CHUNK_SHIFT, CHUNK_SIZE, ChunkPos, Face, Light, LocalPos,
    WorldBounds,
};

use crate::{Chunk, ChunkLight, World};

const SIZE: i32 = CHUNK_SIZE;
const MAX: u8 = Light::MAX;

/// What light needs to know about each block.
#[derive(Clone, Debug)]
pub struct LightTable {
    opaque: Vec<bool>,
    emission: Vec<Light>,
}

impl LightTable {
    pub fn new(blocks: &BlockRegistry) -> Self {
        Self {
            opaque: blocks.iter().map(|(_, def)| def.is_opaque()).collect(),
            emission: blocks
                .iter()
                .map(|(_, def)| def.light.without_sky())
                .collect(),
        }
    }

    fn opaque(&self, id: BlockId) -> bool {
        // Unknown ids show as the solid placeholder block.
        self.opaque.get(id.index()).copied().unwrap_or(true)
    }

    fn emission(&self, id: BlockId) -> Light {
        self.emission.get(id.index()).copied().unwrap_or_default()
    }
}

/// Keeps track of which chunks are lit and spreads light through them.
#[derive(Debug)]
pub struct LightEngine {
    table: LightTable,
    /// Heights of the chunks inside the world.
    world_y: RangeInclusive<i32>,
    lit: HashSet<ChunkPos>,
}

impl LightEngine {
    pub fn new(blocks: &BlockRegistry, bounds: WorldBounds) -> Self {
        Self {
            table: LightTable::new(blocks),
            world_y: (bounds.min_y >> CHUNK_SHIFT)..=(bounds.max_y >> CHUNK_SHIFT),
            lit: HashSet::new(),
        }
    }

    pub fn is_lit(&self, pos: ChunkPos) -> bool {
        self.lit.contains(&pos)
    }

    /// Takes a chunk that was lit elsewhere, like one the server sent with
    /// its light: changes nearby spread through it from now on.
    pub fn mark_lit(&mut self, pos: ChunkPos) {
        self.lit.insert(pos);
    }

    /// The chunk above `pos`, if it has to be lit before `pos` can be.
    pub fn needs_above(&self, pos: ChunkPos) -> Option<ChunkPos> {
        (pos.0.y < *self.world_y.end()).then(|| ChunkPos(pos.0 + IVec3::Y))
    }

    /// Whether `pos` is loaded, not lit yet and the chunk above it is lit.
    pub fn can_light(&self, world: &World, pos: ChunkPos) -> bool {
        world.chunk(pos).is_some()
            && !self.is_lit(pos)
            && self.needs_above(pos).is_none_or(|above| self.is_lit(above))
    }

    /// Lights a loaded chunk: sky light from the chunk above, the light of
    /// its own glowing blocks and the light of its lit neighbours, and
    /// spreads it on into them. Returns the chunks whose light changed.
    ///
    /// The chunk above must be lit already, see [`LightEngine::can_light`].
    pub fn light_chunk(&mut self, world: &mut World, pos: ChunkPos) -> Vec<ChunkPos> {
        debug_assert!(self.can_light(world, pos));
        // Solid rock: nothing gets in and nothing glows.
        if let Some(block) = world.chunk(pos).and_then(Chunk::uniform)
            && self.table.opaque(block)
            && self.table.emission(block).is_dark()
        {
            self.lit.insert(pos);
            return Vec::new();
        }
        let open_air = self.is_open_air(world, pos);
        let mut region = Region::take(world, pos, &self.world_y, |p| {
            p == pos || self.lit.contains(&p)
        });
        let origin = pos.origin().0;
        let mut queue = VecDeque::new();

        if open_air {
            // Sky everywhere; it only needs to spread out of the chunk.
            let chunk = region.chunk_mut(pos).expect("the chunk is in the region");
            chunk.set_light(ChunkLight::uniform(Light::SKY));
            for face in Face::ALL {
                for (a, b) in face_cells(face) {
                    let p = origin + border_cell(face, a, b);
                    if self.lights_a_neighbour(&region, p, Light::SKY) {
                        queue.push_back(p);
                    }
                }
            }
        } else {
            self.light_from_above(&mut region, pos, &mut queue);
        }
        self.add_glowing_blocks(&mut region, pos, &mut queue);

        // Light coming in from lit neighbours.
        for face in Face::ALL {
            let neighbour = pos.offset(face);
            if region.chunk(neighbour).is_none() {
                continue;
            }
            for (a, b) in face_cells(face) {
                // The neighbour's layer that touches this chunk.
                let p = neighbour.origin().0 + border_cell(face.opposite(), a, b);
                if let Some((_, light)) = region.get(p)
                    && self.lights_a_neighbour(&region, p, light)
                {
                    queue.push_back(p);
                }
            }
        }

        self.spread(&mut region, &mut queue);
        self.lit.insert(pos);
        region.mark_changed(pos);
        region.put_back(world)
    }

    /// Whether the chunk is all air with open sky above every column.
    fn is_open_air(&self, world: &World, pos: ChunkPos) -> bool {
        let Some(block) = world.chunk(pos).and_then(Chunk::uniform) else {
            return false;
        };
        if self.table.opaque(block) || !self.table.emission(block).is_dark() {
            return false;
        }
        let Some(above) = self.needs_above(pos) else {
            return true;
        };
        let Some(above) = world.chunk(above) else {
            return false;
        };
        match above.light().as_uniform() {
            Some(light) => light.sky() == MAX,
            None => face_cells(Face::NegY).all(|(a, b)| {
                let local = border_cell(Face::NegY, a, b);
                above.light().get(local_pos(local)).sky() == MAX
            }),
        }
    }

    /// Sky light falling straight down from the chunk above, or from the
    /// open sky at the top of the world, and where it starts to spread.
    fn light_from_above(&self, region: &mut Region, pos: ChunkPos, queue: &mut VecDeque<IVec3>) {
        let origin = pos.origin().0;
        let top_of_world = self.needs_above(pos).is_none();
        let mut open = [[false; SIZE as usize]; SIZE as usize];
        for (z, row) in open.iter_mut().enumerate() {
            for (x, open) in row.iter_mut().enumerate() {
                let above = origin + IVec3::new(x as i32, SIZE, z as i32);
                *open = match region.get(above) {
                    Some((_, light)) => light.sky() == MAX,
                    None => top_of_world,
                };
            }
        }

        // Down every open column until something opaque: `lowest[z][x]` is
        // the lowest block reached, or SIZE where the sky doesn't get in.
        let chunk = region.chunk_mut(pos).expect("the chunk is in the region");
        let mut blocks = vec![BlockId::AIR; ruda_core::CHUNK_VOLUME];
        chunk.copy_to(&mut blocks);
        let lights = chunk.light_mut().values_mut();
        let mut lowest = [[SIZE; SIZE as usize]; SIZE as usize];
        for z in 0..SIZE {
            for x in 0..SIZE {
                if !open[z as usize][x as usize] {
                    continue;
                }
                for y in (0..SIZE).rev() {
                    let index = local_pos(IVec3::new(x, y, z)).index();
                    if self.table.opaque(blocks[index]) {
                        break;
                    }
                    lights[index] = lights[index].with_channel(Light::SKY_CHANNEL, MAX);
                    lowest[z as usize][x as usize] = y;
                }
            }
        }
        region.mark_changed(pos);

        // Sky light spreads sideways where the next column is lit less far
        // down, and down out of the chunk where a column is open to the
        // bottom.
        let transparent = |local: IVec3| !self.table.opaque(blocks[local_pos(local).index()]);
        for z in 0..SIZE {
            for x in 0..SIZE {
                let low = lowest[z as usize][x as usize];
                if low == SIZE {
                    continue;
                }
                for (dx, dz) in [(1, 0), (-1, 0), (0, 1), (0, -1)] {
                    let (nx, nz) = (x + dx, z + dz);
                    let inside = (0..SIZE).contains(&nx) && (0..SIZE).contains(&nz);
                    for y in low..SIZE {
                        if inside && y >= lowest[nz as usize][nx as usize] {
                            break;
                        }
                        let lights_it = if inside {
                            transparent(IVec3::new(nx, y, nz))
                        } else {
                            region.get(origin + IVec3::new(nx, y, nz)).is_some_and(
                                |(block, light)| !self.table.opaque(block) && light.sky() < MAX - 1,
                            )
                        };
                        if lights_it {
                            queue.push_back(origin + IVec3::new(x, y, z));
                        }
                    }
                }
                if low == 0 {
                    let below = origin + IVec3::new(x, -1, z);
                    if region.get(below).is_some_and(|(block, light)| {
                        !self.table.opaque(block) && light.sky() < MAX
                    }) {
                        queue.push_back(origin + IVec3::new(x, 0, z));
                    }
                }
            }
        }
    }

    /// The chunk's own glowing blocks.
    fn add_glowing_blocks(&self, region: &mut Region, pos: ChunkPos, queue: &mut VecDeque<IVec3>) {
        let origin = pos.origin().0;
        let chunk = region.chunk(pos).expect("the chunk is in the region");
        let glows = |block| !self.table.emission(block).is_dark();
        let glowing: Vec<IVec3> = if chunk.palette().iter().any(|&block| glows(block)) {
            LocalPos::all()
                .filter(|&local| glows(chunk.get(local)))
                .map(|local| origin + local.vec())
                .collect()
        } else {
            Vec::new()
        };
        for p in glowing {
            let (block, light) = region.get(p).expect("the chunk is in the region");
            region.set(p, light.max(self.table.emission(block)));
            queue.push_back(p);
        }
    }

    /// Updates the light after the block at `pos` was replaced by `old`'s
    /// successor, already set in `world`. Returns the chunks whose light
    /// changed.
    pub fn block_changed(
        &mut self,
        world: &mut World,
        pos: BlockPos,
        old: BlockId,
    ) -> Vec<ChunkPos> {
        if !self.is_lit(pos.chunk()) {
            return Vec::new();
        }
        let mut region = Region::take(world, pos.chunk(), &self.world_y, |p| self.lit.contains(&p));
        let p = pos.0;
        let (new, mut light) = region.get(p).expect("the block's chunk is in the region");
        let (old_opaque, new_opaque) = (self.table.opaque(old), self.table.opaque(new));
        let (old_emission, new_emission) = (self.table.emission(old), self.table.emission(new));

        // Take away the light that came from or through this block.
        let mut queue = VecDeque::new();
        for channel in 0..Light::CHANNELS {
            let blocks_now = new_opaque && !old_opaque;
            let dimmer = old_emission.channel(channel) > new_emission.channel(channel);
            if (blocks_now || dimmer) && light.channel(channel) > 0 {
                self.remove(&mut region, p, channel, &mut queue);
            }
        }
        light = region.get(p).expect("still there").1;
        if new_opaque {
            light = Light::DARK;
        }
        region.set(p, light.max(new_emission));
        if !new_emission.is_dark() {
            queue.push_back(p);
        }
        // An open block takes the light of its neighbours.
        if !new_opaque {
            for face in Face::ALL {
                let q = p + face.normal();
                if region.get(q).is_some_and(|(_, light)| !light.is_dark()) {
                    queue.push_back(q);
                }
            }
        }
        self.spread(&mut region, &mut queue);
        region.put_back(world)
    }

    /// The chunk was unloaded.
    pub fn forget(&mut self, pos: ChunkPos) {
        self.lit.remove(&pos);
    }

    /// Whether light at `p` would brighten an open neighbour.
    fn lights_a_neighbour(&self, region: &Region, p: IVec3, light: Light) -> bool {
        Face::ALL.iter().any(|&face| {
            region
                .get(p + face.normal())
                .is_some_and(|(block, current)| {
                    !self.table.opaque(block) && passed(light, face).max(current) != current
                })
        })
    }

    /// Spreads light outwards from every block in `queue`.
    fn spread(&self, region: &mut Region, queue: &mut VecDeque<IVec3>) {
        while let Some(p) = queue.pop_front() {
            let Some((_, light)) = region.get(p) else {
                continue;
            };
            for face in Face::ALL {
                let q = p + face.normal();
                let Some((block, current)) = region.get(q) else {
                    continue;
                };
                if self.table.opaque(block) {
                    continue;
                }
                let brighter = passed(light, face).max(current);
                if brighter != current {
                    region.set(q, brighter);
                    queue.push_back(q);
                }
            }
        }
    }

    /// Takes away one channel of the light at `start` and everything that
    /// got its light from there. Blocks lit from elsewhere that border the
    /// darkened area go into `relight`, to spread their light back in.
    fn remove(
        &self,
        region: &mut Region,
        start: IVec3,
        channel: usize,
        relight: &mut VecDeque<IVec3>,
    ) {
        let (_, light) = region.get(start).expect("in the region");
        let mut stack = vec![(start, light.channel(channel))];
        region.set(start, light.with_channel(channel, 0));
        while let Some((p, level)) = stack.pop() {
            for face in Face::ALL {
                let q = p + face.normal();
                let Some((block, light)) = region.get(q) else {
                    continue;
                };
                let theirs = light.channel(channel);
                if theirs == 0 {
                    continue;
                }
                let sky_column = channel == Light::SKY_CHANNEL
                    && face == Face::NegY
                    && level == MAX
                    && theirs == MAX;
                if theirs < level || sky_column {
                    let emission = self.table.emission(block).channel(channel);
                    region.set(q, light.with_channel(channel, emission));
                    if emission > 0 {
                        relight.push_back(q);
                    }
                    stack.push((q, theirs));
                } else {
                    relight.push_back(q);
                }
            }
        }
    }
}

/// The light that crosses into the next block across `face`.
fn passed(light: Light, face: Face) -> Light {
    let mut out = Light::DARK;
    for channel in 0..Light::CHANNELS {
        let level = light.channel(channel);
        let level = if channel == Light::SKY_CHANNEL && face == Face::NegY && level == MAX {
            MAX
        } else {
            level.saturating_sub(1)
        };
        out = out.with_channel(channel, level);
    }
    out
}

/// The 32×32 cells of a chunk face, as coordinates along its two other axes.
fn face_cells(_face: Face) -> impl Iterator<Item = (i32, i32)> {
    (0..SIZE).flat_map(|b| (0..SIZE).map(move |a| (a, b)))
}

/// The cell at (a, b) of a chunk's layer on `face`, relative to its origin.
fn border_cell(face: Face, a: i32, b: i32) -> IVec3 {
    let axis = face.axis();
    let (u, v) = ((axis + 1) % 3, (axis + 2) % 3);
    let mut cell = IVec3::ZERO;
    cell[axis] = if face.is_positive() { SIZE - 1 } else { 0 };
    cell[u] = a;
    cell[v] = b;
    cell
}

/// Chunks taken out of the world while light spreads through them: the
/// 3×3 columns around a chunk, as high as the world. Chunks that aren't
/// taken stop light like walls.
struct Region {
    /// The minimum chunk.
    min: IVec3,
    size: IVec3,
    chunks: Vec<Option<Chunk>>,
    changed: Vec<bool>,
}

impl Region {
    fn take(
        world: &mut World,
        center: ChunkPos,
        world_y: &RangeInclusive<i32>,
        include: impl Fn(ChunkPos) -> bool,
    ) -> Self {
        let min = IVec3::new(center.0.x - 1, *world_y.start(), center.0.z - 1);
        let size = IVec3::new(3, world_y.end() - world_y.start() + 1, 3);
        let mut chunks = Vec::with_capacity((size.x * size.y * size.z) as usize);
        for y in 0..size.y {
            for z in 0..size.z {
                for x in 0..size.x {
                    let pos = ChunkPos(min + IVec3::new(x, y, z));
                    chunks.push(if include(pos) {
                        world.remove_chunk(pos)
                    } else {
                        None
                    });
                }
            }
        }
        let changed = vec![false; chunks.len()];
        Self {
            min,
            size,
            chunks,
            changed,
        }
    }

    /// Returns every chunk to the world and the positions of those whose
    /// light changed.
    fn put_back(self, world: &mut World) -> Vec<ChunkPos> {
        let mut changed = Vec::new();
        let positions: Vec<ChunkPos> = (0..self.chunks.len())
            .map(|index| self.chunk_pos(index))
            .collect();
        for (index, chunk) in self.chunks.into_iter().enumerate() {
            let Some(mut chunk) = chunk else {
                continue;
            };
            let pos = positions[index];
            if self.changed[index] {
                chunk.light_mut().compact();
                changed.push(pos);
            }
            world.insert_chunk(pos, chunk);
        }
        changed
    }

    fn chunk_index(&self, chunk: IVec3) -> Option<usize> {
        let at = chunk - self.min;
        let inside = at.cmpge(IVec3::ZERO).all() && at.cmplt(self.size).all();
        inside.then(|| ((at.y * self.size.z + at.z) * self.size.x + at.x) as usize)
    }

    fn chunk_pos(&self, index: usize) -> ChunkPos {
        let index = index as i32;
        let x = index % self.size.x;
        let z = index / self.size.x % self.size.z;
        let y = index / (self.size.x * self.size.z);
        ChunkPos(self.min + IVec3::new(x, y, z))
    }

    fn chunk(&self, pos: ChunkPos) -> Option<&Chunk> {
        self.chunks[self.chunk_index(pos.0)?].as_ref()
    }

    fn chunk_mut(&mut self, pos: ChunkPos) -> Option<&mut Chunk> {
        let index = self.chunk_index(pos.0)?;
        self.chunks[index].as_mut()
    }

    fn mark_changed(&mut self, pos: ChunkPos) {
        if let Some(index) = self.chunk_index(pos.0) {
            self.changed[index] = true;
        }
    }

    /// The block and light at a world position, if its chunk was taken.
    fn get(&self, p: IVec3) -> Option<(BlockId, Light)> {
        let chunk = self.chunks[self.chunk_index(p >> CHUNK_SHIFT)?].as_ref()?;
        let local = local(p);
        Some((chunk.get(local), chunk.light().get(local)))
    }

    fn set(&mut self, p: IVec3, light: Light) {
        let Some(index) = self.chunk_index(p >> CHUNK_SHIFT) else {
            return;
        };
        if let Some(chunk) = &mut self.chunks[index] {
            chunk.light_mut().set(local(p), light);
            self.changed[index] = true;
        }
    }
}

fn local(p: IVec3) -> LocalPos {
    local_pos(p & (SIZE - 1))
}

fn local_pos(p: IVec3) -> LocalPos {
    LocalPos::new(p.x as u32, p.y as u32, p.z as u32)
}

#[cfg(test)]
mod tests {
    use ruda_core::{Appearance, BlockDef, ContentBuilder, CubeTextures, ResourceId};

    use super::*;

    struct Fixture {
        blocks: BlockRegistry,
        stone: BlockId,
        lamp: BlockId,
        red: BlockId,
        blue: BlockId,
    }

    fn fixture() -> Fixture {
        let mut content = ContentBuilder::new();
        let mut add = |name: &str, light: Light| {
            let id: ResourceId = format!("test:{name}").parse().unwrap();
            let def = BlockDef::new(id.clone(), Appearance::Cube(CubeTextures::all(id)));
            let def = def.emits_light(light.red(), light.green(), light.blue());
            content.add_block(def).unwrap()
        };
        let stone = add("stone", Light::DARK);
        let lamp = add("lamp", Light::rgb(15, 15, 15));
        let red = add("red", Light::rgb(12, 0, 0));
        let blue = add("blue", Light::rgb(0, 0, 12));
        Fixture {
            blocks: content.build().blocks().clone(),
            stone,
            lamp,
            red,
            blue,
        }
    }

    /// Two chunks tall, from y = −32 to 31.
    const BOUNDS: WorldBounds = WorldBounds {
        min_y: -32,
        max_y: 31,
    };

    /// Stone below y = 0, air above, over a 3×3 area of columns.
    fn ground(f: &Fixture) -> World {
        let mut world = World::new();
        for z in -1..=1 {
            for x in -1..=1 {
                world.insert_chunk(ChunkPos::new(x, 0, z), Chunk::filled(BlockId::AIR));
                world.insert_chunk(ChunkPos::new(x, -1, z), Chunk::filled(f.stone));
            }
        }
        world
    }

    fn light_all(engine: &mut LightEngine, world: &mut World) {
        let mut pending: Vec<ChunkPos> = world.chunks().map(|(pos, _)| pos).collect();
        while !pending.is_empty() {
            let before = pending.len();
            pending.retain(|&pos| {
                if engine.can_light(world, pos) {
                    engine.light_chunk(world, pos);
                    false
                } else {
                    true
                }
            });
            assert!(pending.len() < before, "some chunks can never be lit");
        }
    }

    fn light_at(world: &World, x: i32, y: i32, z: i32) -> Light {
        let pos = BlockPos::new(x, y, z);
        world.chunk(pos.chunk()).unwrap().light().get(pos.local())
    }

    fn set(engine: &mut LightEngine, world: &mut World, pos: BlockPos, block: BlockId) {
        let old = world.set_block(pos, block).unwrap();
        engine.block_changed(world, pos, old);
    }

    #[test]
    fn the_sky_lights_open_ground_and_not_below_it() {
        let f = fixture();
        let mut world = ground(&f);
        let mut engine = LightEngine::new(&f.blocks, BOUNDS);
        light_all(&mut engine, &mut world);
        assert_eq!(light_at(&world, 5, 0, 5), Light::SKY);
        assert_eq!(light_at(&world, -20, 31, 7), Light::SKY);
        assert_eq!(light_at(&world, 5, -1, 5), Light::DARK);
        assert_eq!(light_at(&world, 5, -20, 5), Light::DARK);
    }

    #[test]
    fn a_roof_casts_shade_that_fills_in_from_the_sides() {
        let f = fixture();
        let mut world = ground(&f);
        let mut engine = LightEngine::new(&f.blocks, BOUNDS);
        light_all(&mut engine, &mut world);
        set(&mut engine, &mut world, BlockPos::new(3, 4, 3), f.stone);
        // Under the roof block, sky light comes in sideways.
        assert_eq!(light_at(&world, 3, 3, 3).sky(), 14);
        assert_eq!(light_at(&world, 3, 0, 3).sky(), 14);
        assert_eq!(light_at(&world, 4, 0, 3).sky(), 15);

        set(
            &mut engine,
            &mut world,
            BlockPos::new(3, 4, 3),
            BlockId::AIR,
        );
        assert_eq!(light_at(&world, 3, 0, 3), Light::SKY);
    }

    #[test]
    fn a_lamp_lights_a_cave_across_chunk_borders() {
        let f = fixture();
        let mut world = ground(&f);
        let mut engine = LightEngine::new(&f.blocks, BOUNDS);
        // A tunnel along x at y = −10 under the ground, crossing into the
        // next chunk at x = 32.
        for x in 20..40 {
            world.set_block(BlockPos::new(x, -10, 5), BlockId::AIR);
        }
        world.set_block(BlockPos::new(30, -10, 5), f.lamp);
        light_all(&mut engine, &mut world);
        assert_eq!(light_at(&world, 30, -10, 5), Light::rgb(15, 15, 15));
        assert_eq!(light_at(&world, 31, -10, 5), Light::rgb(14, 14, 14));
        assert_eq!(light_at(&world, 35, -10, 5), Light::rgb(10, 10, 10));
        // Stone around the tunnel stays dark.
        assert_eq!(light_at(&world, 31, -9, 5), Light::DARK);

        // Without the lamp the tunnel goes dark.
        set(
            &mut engine,
            &mut world,
            BlockPos::new(30, -10, 5),
            BlockId::AIR,
        );
        assert_eq!(light_at(&world, 35, -10, 5), Light::DARK);
        assert_eq!(light_at(&world, 30, -10, 5), Light::DARK);
    }

    #[test]
    fn colours_mix() {
        let f = fixture();
        let mut world = ground(&f);
        let mut engine = LightEngine::new(&f.blocks, BOUNDS);
        for x in 0..12 {
            world.set_block(BlockPos::new(x, -10, 5), BlockId::AIR);
        }
        world.set_block(BlockPos::new(0, -10, 5), f.red);
        world.set_block(BlockPos::new(11, -10, 5), f.blue);
        light_all(&mut engine, &mut world);
        assert_eq!(light_at(&world, 5, -10, 5), Light::rgb(7, 0, 6));
    }

    /// Changing blocks one by one must end in the same light as lighting
    /// the final world from scratch.
    #[test]
    fn updates_match_lighting_from_scratch() {
        let f = fixture();
        let mut world = ground(&f);
        let mut engine = LightEngine::new(&f.blocks, BOUNDS);
        light_all(&mut engine, &mut world);

        let mut state = 7u32;
        let mut next = |n: i32| {
            state = state.wrapping_mul(1_103_515_245).wrapping_add(12345);
            (state >> 8) as i32 % n
        };
        let choices = [BlockId::AIR, f.stone, f.lamp, f.red, f.blue];
        for _ in 0..300 {
            let pos = BlockPos::new(next(40) - 20, next(24) - 12, next(40) - 20);
            let block = choices[next(choices.len() as i32) as usize];
            set(&mut engine, &mut world, pos, block);
        }

        let mut fresh = World::new();
        for (pos, chunk) in world.chunks() {
            let mut blocks = vec![BlockId::AIR; ruda_core::CHUNK_VOLUME];
            chunk.copy_to(&mut blocks);
            fresh.insert_chunk(pos, Chunk::from_blocks(&blocks));
        }
        let mut fresh_engine = LightEngine::new(&f.blocks, BOUNDS);
        light_all(&mut fresh_engine, &mut fresh);
        for (pos, chunk) in world.chunks() {
            for local in LocalPos::all() {
                assert_eq!(
                    chunk.light().get(local),
                    fresh.chunk(pos).unwrap().light().get(local),
                    "at {:?}",
                    pos.block(local)
                );
            }
        }
    }
}
