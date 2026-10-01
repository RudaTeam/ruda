# Обзор архитектуры

Здесь общая картина. Детали и обоснования — в [ADR](adr/README.md).

## Крейты и зависимости

```mermaid
flowchart TD
    app_client["apps/ruda<br/>клиент + встроенный сервер"]
    app_server["apps/ruda-server<br/>выделенный сервер"]
    base["content/base<br/>базовая игра"]

    app_client --> ruda_client
    app_client --> ruda_server
    app_client --> base
    app_server --> ruda_server
    app_server --> base

    ruda_client --> ruda_render
    ruda_client --> ruda_ui
    ruda_client --> ruda_input
    ruda_client --> ruda_audio
    ruda_client --> ruda_net
    ruda_client --> ruda_sim

    ruda_server --> ruda_net
    ruda_server --> ruda_sim
    ruda_server --> ruda_worldgen
    ruda_server --> ruda_storage

    ruda_net --> ruda_protocol
    ruda_protocol --> ruda_world
    ruda_sim --> ruda_world
    ruda_worldgen --> ruda_world
    ruda_storage --> ruda_world
    ruda_render --> ruda_world
    ruda_world --> ruda_core
    base -.->|только публичный API| ruda_core
```

Граница проходит так: всё, от чего зависит `ruda_server`, собирается без `wgpu`, `winit` и звука.
Это проверяет CI (ADR-0002).

## Процессы и соединения

```mermaid
flowchart LR
    subgraph sp["Одиночная игра — один процесс"]
        c1["Клиент"] <-->|in-memory транспорт| s1["Встроенный сервер"]
    end
    subgraph ds["Выделенный сервер"]
        s2["Сервер"]
    end
    c2["Клиент"] <-->|QUIC| s2
    c3["Клиент"] <-->|QUIC| s2
    friend["Друг в LAN"] <-.->|"QUIC: «открыть для сети»"| s1
```

## Жизненный цикл чанка

```mermaid
sequenceDiagram
    participant T as Тик-поток сервера
    participant IO as IO-поток
    participant SW as Воркеры сервера
    participant CW as Воркеры клиента
    participant M as Главный поток клиента

    T->>IO: загрузить чанк
    IO-->>T: на диске нет
    T->>SW: сгенерировать и осветить
    SW-->>T: готовый чанк
    T->>CW: данные чанка по сети (LZ4)
    CW->>CW: распаковка, мешинг (greedy, квады по 8 байт)
    CW-->>M: готовый меш
    M->>M: загрузка в GPU в рамках бюджета кадра, отрисовка
    T->>IO: сохранить, если изменён (пачкой, zstd)
```

## Потоки

| Где | Поток | Что делает |
|---|---|---|
| Сервер | Тик (20 TPS) | Единственный меняет мир; выполняет расписание систем ECS |
| Сервер | Пул воркеров | Генерация, освещение, сжатие |
| Сервер | IO | Хранилище |
| Оба | Сеть (tokio) | QUIC |
| Клиент | Главный | События окна, ввод, логика 20 Гц с интерполяцией, рендер |
| Клиент | Пул воркеров | Распаковка, мешинг |

В одиночной игре пул воркеров общий (ADR-0012).
