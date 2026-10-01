# Contributing to Ruda

Ruda is in early development and a lot is still moving, so please open an
issue before starting on anything big. A short discussion up front saves
rewriting the pull request later.

## License

Ruda is licensed under [Apache-2.0](LICENSE). By opening a pull request you
agree that your contribution is licensed the same way (section 5 of the
license). You keep the copyright on your work, and there is no CLA to sign.

## Ground rules

- `ruda-server` must build without graphics, windowing or audio crates. CI
  checks its dependency tree.
- Dependencies must use permissive licenses: MIT, Apache-2.0, BSD, ISC, Zlib
  and the like. The exact list lives in `deny.toml` and `cargo deny check`
  enforces it.
- Gameplay has to work within what OpenGL ES 3.0 class hardware can do.
  Features that need more are fine, but only as optional visual extras.
- Code, comments and commit messages are written in English.

## Workflow

1. Branch off `main`.
2. Run the same checks as CI:

   ```sh
   cargo fmt --all --check
   cargo clippy --workspace --all-targets --all-features -- -D warnings
   cargo test --workspace
   cargo deny check    # cargo install cargo-deny --locked
   ```

3. Open a pull request. `main` only takes squash-merged pull requests, and
   the **CI passed** check has to be green.

## Assets

Textures, sounds, models and fonts must be original or available under CC0,
CC-BY (with attribution) or, for fonts, the SIL Open Font License. Record
where every third-party asset came from in `assets/CREDITS.md`.
