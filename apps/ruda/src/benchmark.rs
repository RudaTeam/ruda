//! `--benchmark`: frame times over a fixed view, measured once the world
//! around the player has finished loading.

use std::fmt;
use std::time::{Duration, Instant};

use tracing::warn;

/// How long to wait for the world to load before measuring anyway.
const LOAD_TIMEOUT: Duration = Duration::from_secs(120);

#[derive(Debug)]
pub struct Benchmark {
    duration: Duration,
    state: State,
}

#[derive(Debug)]
enum State {
    Loading {
        since: Instant,
    },
    Measuring {
        last: Instant,
        frames: Vec<Duration>,
        busy: Vec<Duration>,
        /// Per frame the GPU measured: milliseconds per kind of pass.
        gpu: Vec<Vec<(&'static str, f64)>>,
    },
    Done,
}

impl Benchmark {
    pub fn new(duration: Duration) -> Self {
        Self {
            duration,
            state: State::Loading {
                since: Instant::now(),
            },
        }
    }

    /// Call after every frame that reached the screen. `busy` is the time the
    /// frame took without waiting for the display, `loaded` says whether the
    /// world around the player is complete, `gpu` is the GPU's time per kind
    /// of pass for frames measured since the last call. Returns the results
    /// once the measurement is over.
    pub fn frame(
        &mut self,
        now: Instant,
        busy: Duration,
        loaded: bool,
        gpu: Vec<Vec<(&'static str, f64)>>,
    ) -> Option<Report> {
        match &mut self.state {
            State::Loading { since } => {
                let timed_out = now - *since > LOAD_TIMEOUT;
                if timed_out {
                    warn!("the world is still loading, measuring anyway");
                }
                if loaded || timed_out {
                    self.state = State::Measuring {
                        last: now,
                        frames: Vec::new(),
                        busy: Vec::new(),
                        gpu: Vec::new(),
                    };
                }
                None
            }
            State::Measuring {
                last,
                frames,
                busy: busy_times,
                gpu: gpu_times,
            } => {
                frames.push(now - *last);
                busy_times.push(busy);
                gpu_times.extend(gpu);
                *last = now;
                if frames.iter().sum::<Duration>() < self.duration {
                    return None;
                }
                let report = Report::new(
                    std::mem::take(frames),
                    std::mem::take(busy_times),
                    std::mem::take(gpu_times),
                );
                self.state = State::Done;
                Some(report)
            }
            State::Done => None,
        }
    }
}

/// Frame time statistics.
#[derive(Clone, Debug)]
pub struct Report {
    pub frames: usize,
    pub fps: f64,
    /// Time from one frame to the next.
    pub frame_time: Percentiles,
    /// Time a frame kept the CPU busy, without waiting for the display.
    pub busy: Percentiles,
    /// The GPU's time per kind of pass and for the whole frame, where it
    /// can be measured.
    pub gpu: Vec<(&'static str, Percentiles)>,
}

/// Milliseconds.
#[derive(Clone, Copy, Debug)]
pub struct Percentiles {
    pub p50: f64,
    pub p95: f64,
    pub p99: f64,
    pub max: f64,
}

impl Percentiles {
    fn new(mut times: Vec<Duration>) -> Self {
        times.sort();
        let ms = |duration: Duration| duration.as_secs_f64() * 1000.0;
        let percentile = |p: f64| {
            let index = ((times.len() as f64 * p).ceil() as usize).clamp(1, times.len()) - 1;
            ms(times[index])
        };
        Self {
            p50: percentile(0.50),
            p95: percentile(0.95),
            p99: percentile(0.99),
            max: times.last().copied().map_or(0.0, ms),
        }
    }
}

impl fmt::Display for Percentiles {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "p50 {:.2} ms, p95 {:.2} ms, p99 {:.2} ms, max {:.2} ms",
            self.p50, self.p95, self.p99, self.max
        )
    }
}

impl Report {
    fn new(frames: Vec<Duration>, busy: Vec<Duration>, gpu: Vec<Vec<(&'static str, f64)>>) -> Self {
        let total: Duration = frames.iter().sum();
        let ms = |ms: f64| Duration::from_secs_f64(ms / 1000.0);
        let mut kinds: Vec<&'static str> = Vec::new();
        for frame in &gpu {
            for &(kind, _) in frame {
                if !kinds.contains(&kind) {
                    kinds.push(kind);
                }
            }
        }
        let mut gpu_times: Vec<(&'static str, Percentiles)> = kinds
            .into_iter()
            .map(|kind| {
                let times = gpu
                    .iter()
                    .filter_map(|frame| frame.iter().find(|(k, _)| *k == kind))
                    .map(|&(_, time)| ms(time))
                    .collect();
                (kind, Percentiles::new(times))
            })
            .collect();
        if !gpu.is_empty() {
            let frames = gpu
                .iter()
                .map(|frame| ms(frame.iter().map(|&(_, time)| time).sum()))
                .collect();
            gpu_times.push(("whole frame", Percentiles::new(frames)));
        }
        Self {
            frames: frames.len(),
            fps: frames.len() as f64 / total.as_secs_f64(),
            frame_time: Percentiles::new(frames),
            busy: Percentiles::new(busy),
            gpu: gpu_times,
        }
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} frames, {:.1} FPS\nframe time: {}\nbusy:       {}",
            self.frames, self.fps, self.frame_time, self.busy
        )?;
        for (kind, times) in &self.gpu {
            write!(f, "\ngpu, {kind}: {times}")?;
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn waits_for_the_world_then_measures() {
        let mut benchmark = Benchmark::new(Duration::from_millis(100));
        let start = Instant::now();
        let busy = Duration::from_millis(1);
        assert!(benchmark.frame(start, busy, false, Vec::new()).is_none());
        assert!(benchmark.frame(start, busy, true, Vec::new()).is_none());
        let mut now = start;
        let mut report = None;
        for frame in 0..100 {
            // Every tenth frame is slow.
            now += Duration::from_millis(if frame % 10 == 9 { 20 } else { 10 });
            let gpu = vec![vec![("world", 2.0), ("shadows", 1.0)]];
            if let Some(done) = benchmark.frame(now, busy, true, gpu) {
                report = Some(done);
                break;
            }
        }
        let report = report.unwrap();
        assert_eq!(report.frames, 10);
        assert_eq!(report.frame_time.p50, 10.0);
        assert_eq!(report.frame_time.max, 20.0);
        assert_eq!(report.busy.max, 1.0);
        let gpu = |kind| report.gpu.iter().find(|(k, _)| *k == kind).unwrap().1;
        assert_eq!(gpu("world").p50, 2.0);
        assert_eq!(gpu("whole frame").p50, 3.0);
    }
}
