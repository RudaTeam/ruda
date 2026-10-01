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
    /// world around the player is complete. Returns the results once the
    /// measurement is over.
    pub fn frame(&mut self, now: Instant, busy: Duration, loaded: bool) -> Option<Report> {
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
                    };
                }
                None
            }
            State::Measuring {
                last,
                frames,
                busy: busy_times,
            } => {
                frames.push(now - *last);
                busy_times.push(busy);
                *last = now;
                if frames.iter().sum::<Duration>() < self.duration {
                    return None;
                }
                let report = Report::new(std::mem::take(frames), std::mem::take(busy_times));
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
    fn new(frames: Vec<Duration>, busy: Vec<Duration>) -> Self {
        let total: Duration = frames.iter().sum();
        Self {
            frames: frames.len(),
            fps: frames.len() as f64 / total.as_secs_f64(),
            frame_time: Percentiles::new(frames),
            busy: Percentiles::new(busy),
        }
    }
}

impl fmt::Display for Report {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "{} frames, {:.1} FPS\nframe time: {}\nbusy:       {}",
            self.frames, self.fps, self.frame_time, self.busy
        )
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
        assert!(benchmark.frame(start, busy, false).is_none());
        assert!(benchmark.frame(start, busy, true).is_none());
        let mut now = start;
        let mut report = None;
        for frame in 0..100 {
            // Every tenth frame is slow.
            now += Duration::from_millis(if frame % 10 == 9 { 20 } else { 10 });
            if let Some(done) = benchmark.frame(now, busy, true) {
                report = Some(done);
                break;
            }
        }
        let report = report.unwrap();
        assert_eq!(report.frames, 10);
        assert_eq!(report.frame_time.p50, 10.0);
        assert_eq!(report.frame_time.max, 20.0);
        assert_eq!(report.busy.max, 1.0);
    }
}
