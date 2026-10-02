//! The world near the camera on the GPU, for rays towards the sun: which
//! blocks are solid cubes, in a window of chunk columns around the camera
//! that moves with it; which bricks of 8³ blocks hold any, so rays cross
//! open air a brick at a time; and how high the solid blocks in each column
//! of bricks reach, so rays above them skip the column (`ray_shadow` in
//! `world.wgsl`).
//!
//! The window wraps around: a column of chunks sits at its position modulo
//! the window's size, so as the camera moves only the columns coming into
//! the window are written.

use std::collections::HashMap;

use glam::{DVec3, IVec2};
use ruda_core::{CHUNK_SIZE, ChunkPos, WorldBounds};

/// Columns of chunks across the window, along x and z; the camera is in the
/// eighth.
const WINDOW_CHUNKS: i32 = 16;
/// The window's width in blocks (`RAY_WINDOW` in `world.wgsl`).
pub(crate) const RAY_WINDOW: u32 = (WINDOW_CHUNKS * CHUNK_SIZE) as u32;
/// Layers of chunks the window holds: worlds up to 512 blocks tall.
const LAYERS: u32 = 16;
/// Blocks along each side of a brick (`RAY_BRICK` in `world.wgsl`).
const BRICK: usize = 8;
const BRICKS_PER_CHUNK: usize = CHUNK_SIZE as usize / BRICK;

/// A chunk's solid cubes: a column of 32 bits up y for each x and z, entry
/// `z * 32 + x`; see `ChunkMesh::solids`.
pub(crate) type Solids = Box<[u32; (CHUNK_SIZE * CHUNK_SIZE) as usize]>;

pub(crate) struct RayWorld {
    solids: HashMap<ChunkPos, Solids>,
    /// How high solid blocks reach in each column of chunks of the window,
    /// from its lowest block, by where it sits in the window.
    column_tops: [u16; (WINDOW_CHUNKS * WINDOW_CHUNKS) as usize],
    /// On the GPU while rays are on.
    textures: Option<RayTextures>,
    /// The window's first column of chunks and lowest layer, once placed.
    placed: Option<(IVec2, i32)>,
}

struct RayTextures {
    /// A word of 32 blocks up y for each x and z, a layer per layer of
    /// chunks: `RAY_WINDOW` × `LAYERS` × `RAY_WINDOW`.
    solids: wgpu::Texture,
    /// 1 for each brick that holds a solid block.
    bricks: wgpu::Texture,
    /// How high solid blocks reach in each column of bricks, from the
    /// window's lowest block; 0 for none.
    tops: wgpu::Texture,
}

/// What the shader reads: the textures, or tiny empty ones while rays are
/// off.
pub(crate) struct RayViews {
    pub solids: wgpu::TextureView,
    pub bricks: wgpu::TextureView,
    pub tops: wgpu::TextureView,
}

impl RayWorld {
    pub(crate) fn new() -> Self {
        Self {
            solids: HashMap::new(),
            column_tops: [0; (WINDOW_CHUNKS * WINDOW_CHUNKS) as usize],
            textures: None,
            placed: None,
        }
    }

    /// Puts the textures on the GPU, or takes them off, and gives the views
    /// to read.
    pub(crate) fn set_enabled(&mut self, device: &wgpu::Device, enabled: bool) -> RayViews {
        let texture = |label, size: (u32, u32, u32), format| {
            device.create_texture(&wgpu::TextureDescriptor {
                label: Some(label),
                size: wgpu::Extent3d {
                    width: size.0,
                    height: size.1,
                    depth_or_array_layers: size.2,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: if size.2 == 1 && format == wgpu::TextureFormat::R16Uint {
                    wgpu::TextureDimension::D2
                } else {
                    wgpu::TextureDimension::D3
                },
                format,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            })
        };
        let (solids, bricks, tops) = if enabled {
            let bricks = RAY_WINDOW / BRICK as u32;
            (
                texture(
                    "ray solids",
                    (RAY_WINDOW, LAYERS, RAY_WINDOW),
                    wgpu::TextureFormat::R32Uint,
                ),
                texture(
                    "ray bricks",
                    (bricks, LAYERS * BRICKS_PER_CHUNK as u32, bricks),
                    wgpu::TextureFormat::R8Uint,
                ),
                texture(
                    "ray tops",
                    (bricks, bricks, 1),
                    wgpu::TextureFormat::R16Uint,
                ),
            )
        } else {
            (
                texture("ray solids", (1, 1, 1), wgpu::TextureFormat::R32Uint),
                texture("ray bricks", (1, 1, 1), wgpu::TextureFormat::R8Uint),
                texture("ray tops", (1, 1, 1), wgpu::TextureFormat::R16Uint),
            )
        };
        let views = RayViews {
            solids: solids.create_view(&Default::default()),
            bricks: bricks.create_view(&Default::default()),
            tops: tops.create_view(&Default::default()),
        };
        self.textures = enabled.then_some(RayTextures {
            solids,
            bricks,
            tops,
        });
        self.placed = None;
        views
    }

    /// A chunk's solid cubes changed: `None` if it has none or is gone.
    pub(crate) fn set(&mut self, queue: &wgpu::Queue, pos: ChunkPos, solids: Option<Solids>) {
        let changed = match solids {
            Some(solids) => self
                .solids
                .insert(pos, solids)
                .is_none_or(|old| old != self.solids[&pos]),
            None => self.solids.remove(&pos).is_some(),
        };
        if changed && self.in_window(pos) {
            self.write(queue, pos);
            self.write_tops(queue, pos.0.x, pos.0.z);
        }
    }

    pub(crate) fn clear(&mut self) {
        self.solids.clear();
        self.column_tops.fill(0);
        self.placed = None;
    }

    /// Moves the window with the camera, writing the columns that come into
    /// it. Returns the first block of the window along x and z, its lowest
    /// block, and its height in blocks, for the shader; all zero while rays
    /// are off.
    pub(crate) fn update(
        &mut self,
        queue: &wgpu::Queue,
        camera: DVec3,
        bounds: WorldBounds,
    ) -> [i32; 4] {
        if self.textures.is_none() {
            return [0; 4];
        }
        let column = IVec2::new(
            (camera.x / f64::from(CHUNK_SIZE)).floor() as i32,
            (camera.z / f64::from(CHUNK_SIZE)).floor() as i32,
        );
        let first = column - IVec2::splat(WINDOW_CHUNKS / 2 - 1);
        let bottom = bounds.min_y.div_euclid(CHUNK_SIZE);
        let old = self.placed.replace((first, bottom));
        if old != Some((first, bottom)) {
            let was_in = |x: i32, z: i32| {
                old.is_some_and(|(old_first, old_bottom)| {
                    old_bottom == bottom
                        && (old_first.x..old_first.x + WINDOW_CHUNKS).contains(&x)
                        && (old_first.y..old_first.y + WINDOW_CHUNKS).contains(&z)
                })
            };
            for z in first.y..first.y + WINDOW_CHUNKS {
                for x in first.x..first.x + WINDOW_CHUNKS {
                    if !was_in(x, z) {
                        for y in bottom..bottom + LAYERS as i32 {
                            self.write(queue, ChunkPos::new(x, y, z));
                        }
                        self.write_tops(queue, x, z);
                    }
                }
            }
        }
        // Above the highest solid block in the window, rays meet nothing.
        let top = self.column_tops.iter().copied().max().unwrap_or(0);
        [
            first.x * CHUNK_SIZE,
            bottom * CHUNK_SIZE,
            first.y * CHUNK_SIZE,
            i32::from(top).max(1),
        ]
    }

    fn in_window(&self, pos: ChunkPos) -> bool {
        self.placed.is_some_and(|(first, bottom)| {
            (first.x..first.x + WINDOW_CHUNKS).contains(&pos.0.x)
                && (first.y..first.y + WINDOW_CHUNKS).contains(&pos.0.z)
                && (bottom..bottom + LAYERS as i32).contains(&pos.0.y)
        })
    }

    /// Writes a chunk's solid cubes and bricks where the window holds them.
    fn write(&self, queue: &wgpu::Queue, pos: ChunkPos) {
        let (Some(textures), Some((_, bottom))) = (&self.textures, self.placed) else {
            return;
        };
        let empty = [0u32; (CHUNK_SIZE * CHUNK_SIZE) as usize];
        let solids = self.solids.get(&pos).map_or(&empty, |solids| &**solids);
        let at = |c: i32| (c * CHUNK_SIZE).rem_euclid(RAY_WINDOW as i32) as u32;
        let layer = (pos.0.y - bottom) as u32;
        let size = CHUNK_SIZE as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &textures.solids,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: at(pos.0.x),
                    y: layer,
                    z: at(pos.0.z),
                },
                aspect: wgpu::TextureAspect::All,
            },
            &words_bytes(solids),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(size * 4),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: size,
                height: 1,
                depth_or_array_layers: size,
            },
        );
        let bricks = bricks(solids);
        let per = BRICKS_PER_CHUNK as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &textures.bricks,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: at(pos.0.x) / BRICK as u32,
                    y: layer * per,
                    z: at(pos.0.z) / BRICK as u32,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &bricks,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(per),
                rows_per_image: Some(per),
            },
            wgpu::Extent3d {
                width: per,
                height: per,
                depth_or_array_layers: per,
            },
        );
    }
}

impl RayWorld {
    /// Writes how high solid blocks reach in each column of bricks of a
    /// column of chunks, from the window's lowest block.
    fn write_tops(&mut self, queue: &wgpu::Queue, x: i32, z: i32) {
        let (Some(textures), Some((_, bottom))) = (&self.textures, self.placed) else {
            return;
        };
        let size = CHUNK_SIZE as usize;
        let mut tops = [0u16; BRICKS_PER_CHUNK * BRICKS_PER_CHUNK];
        for layer in 0..LAYERS as i32 {
            let Some(solids) = self.solids.get(&ChunkPos::new(x, bottom + layer, z)) else {
                continue;
            };
            for (index, top) in tops.iter_mut().enumerate() {
                let (bx, bz) = (index % BRICKS_PER_CHUNK, index / BRICKS_PER_CHUNK);
                let highest = (bz * BRICK..(bz + 1) * BRICK)
                    .flat_map(|z| (bx * BRICK..(bx + 1) * BRICK).map(move |x| z * size + x))
                    .map(|i| 32 - solids[i].leading_zeros())
                    .max()
                    .unwrap_or(0);
                if highest > 0 {
                    *top = (*top).max((layer * CHUNK_SIZE) as u16 + highest as u16);
                }
            }
        }
        let slot = |c: i32| c.rem_euclid(WINDOW_CHUNKS) as usize;
        self.column_tops[slot(z) * WINDOW_CHUNKS as usize + slot(x)] =
            tops.iter().copied().max().unwrap_or(0);
        let per = BRICKS_PER_CHUNK as u32;
        let at = |c: i32| (c * per as i32).rem_euclid((RAY_WINDOW / BRICK as u32) as i32) as u32;
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &textures.tops,
                mip_level: 0,
                origin: wgpu::Origin3d {
                    x: at(x),
                    y: at(z),
                    z: 0,
                },
                aspect: wgpu::TextureAspect::All,
            },
            &tops
                .iter()
                .flat_map(|top| top.to_le_bytes())
                .collect::<Vec<u8>>(),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(per * 2),
                rows_per_image: Some(per),
            },
            wgpu::Extent3d {
                width: per,
                height: per,
                depth_or_array_layers: 1,
            },
        );
    }
}

/// Which bricks of a chunk hold a solid cube: 1 or 0, x fastest, then y,
/// then z.
fn bricks(solids: &[u32; (CHUNK_SIZE * CHUNK_SIZE) as usize]) -> [u8; BRICKS_PER_CHUNK.pow(3)] {
    let size = CHUNK_SIZE as usize;
    let mut bricks = [0u8; BRICKS_PER_CHUNK.pow(3)];
    for (index, brick) in bricks.iter_mut().enumerate() {
        let (bx, by, bz) = (
            index % BRICKS_PER_CHUNK,
            index / BRICKS_PER_CHUNK % BRICKS_PER_CHUNK,
            index / (BRICKS_PER_CHUNK * BRICKS_PER_CHUNK),
        );
        let rows = 0xffu32 << (by * BRICK);
        let any = (bz * BRICK..(bz + 1) * BRICK)
            .any(|z| (bx * BRICK..(bx + 1) * BRICK).any(|x| solids[z * size + x] & rows != 0));
        *brick = u8::from(any);
    }
    bricks
}

/// The bytes of words, in the GPU's (little-endian) order.
fn words_bytes(words: &[u32]) -> Vec<u8> {
    words.iter().flat_map(|word| word.to_le_bytes()).collect()
}
