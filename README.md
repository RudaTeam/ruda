# Ruda

Ruda is a voxel sandbox game in the spirit of Minecraft, written in Rust.

What we're aiming for:

- multiplayer from the start, with single-player running on a built-in local server;
- smooth play on weak hardware, with a Raspberry Pi 4 as the low bar;
- a modding API that can carry big tech mods: machines, power networks, their own UIs.

It's very early. Right now the client opens a window and clears it with wgpu,
and the dedicated server does nothing. There's nothing to play yet.

## Building

Install Rust with [rustup](https://rustup.rs). The toolchain pinned in
`rust-toolchain.toml` is picked up automatically.

```sh
cargo run -p ruda          # client
cargo run -p ruda-server   # dedicated server
```

The client takes a few options:

- `--gpu-backend <auto|vulkan|metal|dx12|gl>` selects the graphics API. You can
  also set it with `RUDA_GPU_BACKEND`. `auto` tries Vulkan, Metal and DX12
  before falling back to OpenGL.
- `--exit-after-frames <N>` quits after N frames have been shown. CI uses it
  as a smoke test.

Logging is controlled with `RUST_LOG`, for example `RUST_LOG=debug`. To profile
with [Tracy](https://github.com/wolfpld/tracy), build with `--features tracy`.

## Platforms

CI builds Ruda for Windows, macOS and Linux, on both x86_64 and ARM. The
renderer needs a GPU with Vulkan, Metal or DX12, or at least OpenGL 3.3 /
OpenGL ES 3.0.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

Apache-2.0, see [LICENSE](LICENSE). If you fork Ruda, keep the attribution
from [NOTICE](NOTICE).
