//! How long the GPU spends on each pass of a frame, where the graphics card
//! can say (timestamp queries): for the benchmark.
//!
//! Each pass writes the GPU's clock as it starts and ends. At the end of a
//! frame the readings are copied into a buffer that is read a frame or two
//! later, so measuring never makes the CPU wait for the GPU.

use std::sync::Arc;
use std::sync::atomic::{AtomicU8, Ordering};

/// The most passes measured in a frame.
const MOST_PASSES: u32 = 16;
/// Frames whose readings can be on their way back at once.
const READBACKS: usize = 3;
/// Readings kept until someone asks for them.
const KEPT: usize = 600;

const FREE: u8 = 0;
const COPIED: u8 = 1;
const MAPPING: u8 = 2;
const MAPPED: u8 = 3;

pub(crate) struct GpuTimer {
    queries: wgpu::QuerySet,
    resolved: wgpu::Buffer,
    readbacks: Vec<Readback>,
    /// This frame's passes, in order.
    passes: Vec<&'static str>,
    /// Nanoseconds per tick of the GPU's clock.
    period: f64,
    /// Milliseconds per kind of pass, for frames read back since asked.
    finished: Vec<Vec<(&'static str, f64)>>,
}

struct Readback {
    buffer: wgpu::Buffer,
    passes: Vec<&'static str>,
    state: Arc<AtomicU8>,
}

impl GpuTimer {
    /// `None` where the device can't measure passes.
    pub(crate) fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            return None;
        }
        let size = u64::from(MOST_PASSES * 2) * 8;
        let buffer = |label, usage| {
            device.create_buffer(&wgpu::BufferDescriptor {
                label: Some(label),
                size,
                usage,
                mapped_at_creation: false,
            })
        };
        Some(Self {
            queries: device.create_query_set(&wgpu::QuerySetDescriptor {
                label: Some("pass times"),
                ty: wgpu::QueryType::Timestamp,
                count: MOST_PASSES * 2,
            }),
            resolved: buffer(
                "pass times",
                wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            ),
            readbacks: (0..READBACKS)
                .map(|_| Readback {
                    buffer: buffer(
                        "pass times, read back",
                        wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    ),
                    passes: Vec::new(),
                    state: Arc::new(AtomicU8::new(FREE)),
                })
                .collect(),
            passes: Vec::new(),
            period: f64::from(queue.get_timestamp_period()),
            finished: Vec::new(),
        })
    }

    pub(crate) fn begin_frame(&mut self) {
        self.passes.clear();
    }

    /// Where a pass of this kind writes its start and end; `None` past the
    /// most passes a frame measures.
    pub(crate) fn pass(
        &mut self,
        kind: &'static str,
    ) -> Option<wgpu::RenderPassTimestampWrites<'_>> {
        let index = self.passes.len() as u32;
        if index >= MOST_PASSES {
            return None;
        }
        self.passes.push(kind);
        Some(wgpu::RenderPassTimestampWrites {
            query_set: &self.queries,
            beginning_of_pass_write_index: Some(index * 2),
            end_of_pass_write_index: Some(index * 2 + 1),
        })
    }

    /// Copies this frame's readings out, if a buffer is free for them.
    pub(crate) fn end_frame(&mut self, encoder: &mut wgpu::CommandEncoder) {
        let count = self.passes.len() as u32 * 2;
        let Some(readback) = self
            .readbacks
            .iter_mut()
            .find(|readback| readback.state.load(Ordering::Acquire) == FREE)
        else {
            return;
        };
        if count == 0 {
            return;
        }
        encoder.resolve_query_set(&self.queries, 0..count, &self.resolved, 0);
        encoder.copy_buffer_to_buffer(&self.resolved, 0, &readback.buffer, 0, u64::from(count) * 8);
        readback.passes = std::mem::take(&mut self.passes);
        readback.state.store(COPIED, Ordering::Release);
    }

    /// Once the frame is submitted, asks for its readings.
    pub(crate) fn after_submit(&mut self) {
        for readback in &mut self.readbacks {
            if readback.state.load(Ordering::Acquire) == COPIED {
                readback.state.store(MAPPING, Ordering::Release);
                let state = Arc::clone(&readback.state);
                readback
                    .buffer
                    .slice(..)
                    .map_async(wgpu::MapMode::Read, move |result| {
                        state.store(
                            if result.is_ok() { MAPPED } else { FREE },
                            Ordering::Release,
                        );
                    });
            }
        }
    }

    /// Reads back the frames that have arrived.
    pub(crate) fn collect(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::PollType::Poll);
        for readback in &mut self.readbacks {
            if readback.state.load(Ordering::Acquire) != MAPPED {
                continue;
            }
            {
                let Ok(bytes) = readback.buffer.slice(..).get_mapped_range() else {
                    continue;
                };
                let ticks: Vec<u64> = bytes
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|&tick| u64::from_le_bytes(tick))
                    .collect();
                let mut kinds: Vec<(&'static str, f64)> = Vec::new();
                for (i, &kind) in readback.passes.iter().enumerate() {
                    let ns = ticks[i * 2 + 1].saturating_sub(ticks[i * 2]) as f64 * self.period;
                    match kinds.iter_mut().find(|(k, _)| *k == kind) {
                        Some((_, ms)) => *ms += ns / 1e6,
                        None => kinds.push((kind, ns / 1e6)),
                    }
                }
                if self.finished.len() == KEPT {
                    self.finished.remove(0);
                }
                self.finished.push(kinds);
            }
            readback.buffer.unmap();
            readback.state.store(FREE, Ordering::Release);
        }
    }

    /// The frames read back since last asked: milliseconds per kind of pass.
    pub(crate) fn take(&mut self) -> Vec<Vec<(&'static str, f64)>> {
        std::mem::take(&mut self.finished)
    }
}
