//! Headless dedicated server (ADR-0006).
//!
//! Must build without graphics, windowing or audio (ADR-0002): CI checks that
//! its dependency tree contains neither `wgpu` nor `winit`.

fn main() {
    let filter = tracing_subscriber::EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info"));
    tracing_subscriber::fmt().with_env_filter(filter).init();
    tracing::info!(
        version = env!("CARGO_PKG_VERSION"),
        "nothing to serve yet: the game server arrives in M2"
    );
}
