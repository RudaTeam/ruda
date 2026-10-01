# Ruda

Воксельная песочница в духе Minecraft на Rust: свой контент, мультиплеер, работа на слабом железе
вплоть до Raspberry Pi и моддинг уровня Industrial Craft 2.

> **Статус:** веха M0 (фундамент): окно, которое заливается цветом через wgpu. Игры пока нет.

## Сборка и запуск

Нужен только [rustup](https://rustup.rs): нужная версия Rust поставится сама из `rust-toolchain.toml`.

```sh
cargo run -p ruda                         # клиент
cargo run -p ruda -- --gpu-backend gl     # выбрать API: auto, vulkan, metal, dx12, gl
cargo run -p ruda --features tracy        # с профилировщиком Tracy
cargo run -p ruda-server                  # выделенный сервер (пока заглушка)
```

- Бэкенд можно задать и переменной `RUDA_GPU_BACKEND`.
- Подробность логов — через `RUST_LOG`, например `RUST_LOG=debug`.
- `--exit-after-frames N` — отрисовать N кадров и выйти (для смоук-тестов).

Перед коммитом — то же, что проверяет CI:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets --all-features -- -D warnings
cargo test --workspace
cargo deny check          # cargo install cargo-deny --locked
```

## Документация

- [Видение проекта](docs/vision.md) — цели, принципы и то, что в цели не входит
- [Обзор архитектуры](docs/architecture.md) — крейты, процессы, жизненный цикл чанка
- [Дорожная карта](docs/roadmap.md)
- [Архитектурные решения (ADR)](docs/adr/README.md)
- [Глоссарий](docs/glossary.md)

## Ключевые решения

| Область | Решение |
|---|---|
| Графика | Свой движок на wgpu + winit; базовый уровень — OpenGL ES 3.0 |
| Платформы | Windows, macOS, Linux x86_64 и ARM; мобильные позже |
| Сеть | Авторитарный сервер, QUIC (quinn); одиночная игра идёт на встроенном сервере |
| Мир | Кубические чанки 32³ с палитрой *(предложено)* |
| Моддинг | Механизм отложен, требования уровня IC2 заложены в архитектуру |
| Лицензия | Apache-2.0: форки любые, с сохранением атрибуции из `NOTICE` |

## Лицензия

© 2026 RudaTeam / F4 Studio. Распространяется под лицензией [Apache-2.0](LICENSE).
Форки и переиспользование разрешены при условии сохранения атрибуции из файла [NOTICE](NOTICE).
