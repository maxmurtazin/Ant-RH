# Ant-RH — архитектура репозитория

Ниже — обзор модулей и типичного потока данных (полный пайплайн через [`run_full_pipeline.sh`](../../run_full_pipeline.sh)). **Таблицы по функциям и методам** вынесены в [code-reference.md](code-reference.md).

## Структура каталогов и роли

```mermaid
flowchart TB
  subgraph root["Корень репозитория"]
    R_README["README.md"]
    R_REQ["requirements.txt"]
    R_PIPE["run_full_pipeline.sh"]
    R_TRUTH["fractal_dtes_aco_zeta_all_zeros_scan.py"]
    R_METRICS["fractal_dtes_aco_zeta_metrics.py"]
    R_VISUAL["fractal_dtes_aco_zeta_visual.py"]
    R_ETA["fractal_dtes_aco_zeta_eta.py"]
    R_CC["fractal_dtes_aco_zeta_crosschannel.py"]
    R_RUN_V2["run_crosschannel_live_v2.py"]
    R_RUN_OLD["run_crosschannel_live.py / run_crosschannel_fixed.py"]
    R_FIG["run_with_result_figures.py"]
  end

  subgraph core["core/"]
    C_EXPLORE["fractal_dtes_crosschannel_explore_eta_clean.py\n(дефолт DTES в пайплайне)"]
  end

  subgraph hybrid["hybrid/"]
    H_SCAN["hybrid_dtes_guided_scan.py"]
  end

  subgraph refinement["refinement/"]
    REF_COLORS["colored_ants_engine.py"]
    REF_GAP["gap_detector.py"]
    REF_ROADS["dynamic_roads.py"]
  end

  subgraph validation["validation/"]
    V_DIST["distance_analysis.py"]
    V_FIG["figures_from_results.py"]
    V_VAL["validate_zeros_and_spacing_eta.py"]
    V_TRUTH["fractal_dtes_aco_zeta_all_zeros_scan.py"]
  end

  subgraph runs["runs/"]
    RUN_OUT["Артефакты экспериментов\n(truth, JSON, отчёты)"]
  end

  subgraph archive["archive_old/"]
    ARC["Исторические снимки\n(не часть поддерживаемого пайплайна)"]
  end

  R_PIPE --> core
  R_PIPE --> hybrid
  R_PIPE --> refinement
  R_PIPE --> validation
  R_PIPE --> runs
  R_METRICS --> R_VISUAL
  R_METRICS --> R_ETA
  R_METRICS --> R_CC
```

## Поток полного пайплайна (`run_full_pipeline.sh`)

```mermaid
flowchart TD
  S0["Шаг 0: эталонные нули\nfractal_dtes_aco_zeta_all_zeros_scan.py"]
  S1["Шаг 1: DTES core\ncore/fractal_dtes_crosschannel_explore_eta_clean.py"]
  S2["Шаг 2: цветные муравьи\nrefinement/colored_ants_engine.py"]
  S3["Шаг 3: детектор разрывов\nrefinement/gap_detector.py"]
  S4a["Шаг 4: distance — core DTES\nvalidation/distance_analysis.py"]
  S4b["Шаг 5: distance — colored\nvalidation/distance_analysis.py"]
  S6["Шаг 6: гибридное сканирование\nhybrid/hybrid_dtes_guided_scan.py"]
  S7["Шаг 7: distance — hybrid\nvalidation/distance_analysis.py"]
  OUT["runs/run_* / RUN_SUMMARY.md"]

  S0 -->|truth.json| S1
  S1 -->|dtes_candidates.json, edge-aware, metrics| S2
  S2 -->|colored_candidates.json| S3
  S1 --> S4a
  S2 --> S4b
  S2 --> S6
  S6 -->|hybrid_colored.json| S7
  S0 --> S4a
  S0 --> S4b
  S0 --> S7
  S4a --> OUT
  S4b --> OUT
  S6 --> OUT
  S7 --> OUT
  S3 --> OUT
```

## Альтернативный трек: полный Fractal + ACO (вне `run_full_pipeline.sh`)

```mermaid
flowchart LR
  M["fractal_dtes_aco_zeta_metrics.py"]
  V2["run_crosschannel_live_v2.py\n(предпочтительный CLI)"]
  M --> V2
  M --> CC["fractal_dtes_aco_zeta_crosschannel.py"]
  CC --> ETA["fractal_dtes_aco_zeta_eta.py"]
  M --> VIS["fractal_dtes_aco_zeta_visual.py"]
```

## `fractal_dtes_aco_zeta_metrics.py` — модель данных

```mermaid
classDiagram
  direction TB
  class ZetaSearchConfig {
    +t_min: float
    +t_max: float
    +n_grid: int
    +tree_depth: int
    +n_ants: int
    +n_iterations: int
    +ant_types: tuple
    +mp_dps: int
    +top_candidate_nodes: int
    +__post_init__()
  }
  class FractalNode {
    +node_id: int
    +level: int
    +interval: tuple
    +point_ids: list
    +signature: ndarray
    +energy: float
    +min_log_abs: float
    +center(): float
    +width(): float
  }
  class AntPath {
    +node_ids: list
    +score: float
    +agent_type: str
  }
  class DTESAnt {
    +agent_id: int
    +agent_type: str
    +memory_node_id: optional
  }
  class FractalDTESACOZeta {
    -nodes: dict
    -pheromones: dict
    -pheromone_channels: dict
    +run() list
  }
  FractalDTESACOZeta --> ZetaSearchConfig: cfg
  FractalDTESACOZeta "1" o-- "many" FractalNode
  FractalDTESACOZeta ..> AntPath: создаёт
  FractalDTESACOZeta ..> DTESAnt: make_ant
```

> `ZetaSearchConfig` — большой набор весов (энергия, барьеры, pheromone, cross-channel, `w_*`, `lambda_*`, `gamma_*` и т.д.); в диаграмме перечислены только «якорные» поля.

## `FractalDTESACOZeta` — методы по этапам (метрики + ACO)

```mermaid
classDiagram
  class FractalDTESACOZeta {
    <<основной класс>>
    +__init__(cfg)
    +run() List
    +evaluate_grid()
    +compute_multiscale_features()
    +build_dyadic_tree()
    +aggregate_node_statistics()
    +compute_node_energy(node)
    +build_graph()
    +add_undirected_edge(a, b)
    +initialize_pheromones()
    +run_aco()
    +sample_ant_path(start, ant)
    +make_ant(ant_id)
    +reinforce_pheromones(paths)
    +evaporate_pheromones()
    +rank_candidate_nodes()
    +refine_candidates(nodes)
    +local_refinement(interval)
    +verify_zero_candidate(t)
    +merge_close_candidates(cands, tol)
    +evaluate_path(node_ids, agent_type)
    +mixed_pheromone(agent, a, b)
    +cross_channel_pheromone(agent, a, b)
  }
  note for FractalDTESACOZeta "Служебные/стат.: _make_node, _split_node_recursive, path_coherence, …; внизу модуля — _gini, _safe_corr, … (полный список: code-reference.md)"
```

Полная таблица категорий/методов `FractalDTESACOZeta` и заметка о дубликате в `fractal_dtes_aco_zeta_crosschannel.py` — [code-reference: FractalDTESACOZeta](code-reference.md#fractaldtesacozeta).

## Наследование: live runner, визуализация, ETA

```mermaid
classDiagram
  direction TB
  class FractalDTESACOZeta
  class LiveFractalDTESACOZeta
  class StageTimer
  class VisualFractalDTESACOZeta
  class ETALogger
  class ETAFractalDTESACOZeta
  FractalDTESACOZeta <|-- LiveFractalDTESACOZeta : run_crosschannel_live_v2
  FractalDTESACOZeta <|-- VisualFractalDTESACOZeta : visual
  VisualFractalDTESACOZeta <|-- ETAFractalDTESACOZeta : eta
  LiveFractalDTESACOZeta o-- StageTimer
  ETAFractalDTESACOZeta o-- ETALogger
  class LiveFractalDTESACOZeta {
    +__init__(cfg, timer, ...)
    +time_stage(name, fn, ...)
    +run() list
    +run_aco()
    +current_best_energy()
  }
  class VisualFractalDTESACOZeta {
    +plot_energy_landscape()
    +plot_tree_energy_by_level()
    +plot_pheromone_distribution()
    +plot_channel_summary()
    +plot_ramsey_coloring()
    +plot_agent_path_scores()
    +generate_visualizations()
    +visual_run()
  }
  class ETALogger {
    +emit(stage, message, data)
  }
  class ETAFractalDTESACOZeta {
    +run()
    +run_aco()
    +refine_candidates(nodes)
    +plot_aco_history()
    +plot_all()
  }
  class StageTimer {
    +log(stage, message)
  }
```

`run_crosschannel_live_v2`: `make_edge_anchors`, `clean_candidates`, `save_candidates`, `save_metrics`, `main`.

`run_crosschannel_live` (v1) — близкие `StageTimer` / `LiveFractalDTESACOZeta` / `add_edge_anchors` (другой JSON метрик).

## `core/fractal_dtes_crosschannel_explore_eta_clean.py` — сетка без дерева/ACO

Сетка, отбор кандидатов и JSON (класс `Timer`, перечень функций): [code-reference: Core module](code-reference.md#core-module).

```mermaid
flowchart LR
  A[evaluate_grid] --> B[build_candidate_pool]
  B --> C[select_with_exploration]
  C --> D[refine_candidates]
  D --> E[merge_close]
  E --> F[make_edge_anchors]
  F --> G[save_candidate_json / save_edgeaware_json]
  G --> H[json_dump metrics]
```

## `refinement/colored_ants_engine.py`

```mermaid
classDiagram
  class ColoredAnt
  class RoadEdge
  class ColoredAntConfig
  class ColoredGroupedAntEngine
  ColoredAntConfig *-- ColoredGroupedAntEngine
  ColoredAnt ..> ColoredGroupedAntEngine : run_group
  ColoredGroupedAntEngine o-- RoadEdge : edges
  class ColoredAnt {
    +ant_id: int
    +color: str
    +memory: list
  }
  class RoadEdge {
    +i: int
    +j: int
    +distance: float
    +barrier: float
  }
  class ColoredAntConfig {
    +groups: int
    +ants_per_group: int
    +iterations_per_group: int
    +target_count: int
    +evaporation: float
    +k_neighbors: int
  }
  class ColoredGroupedAntEngine {
    +_build_roads()
    +road_value(color, i, j)
    +color_bonus()
    +gap_bonus(t)
    +choose_next(ant, current)
    +path_score()
    +reinforce() / evaporate()
    +run_group(color, idx)
    +run() list
  }
```

Модульные: `load_points`, `save_candidates`, `main`.

`refinement/dynamic_roads.py` — только **текстовая спецификация** формул R_t^color; исполняемой логики нет (см. комментарий в файле).  
`refinement/gap_detector.py`: `load_ts`, `main` — детекция больших зазоров в отсортированных кандидатах.

## `hybrid/hybrid_dtes_guided_scan.py`

Окна, скан по Hardy Z, класс `ETA`, CLI: [code-reference: Hybrid module](code-reference.md#hybrid-module).

## `validation/`

- Сравнение с эталоном и эталонный скан: [Validation distance](code-reference.md#validation-distance) (включает цепочку `distance_analysis` и `all_zeros_scan`).
- Рисунки из JSON-метрик: [Validation figures](code-reference.md#validation-figures).
- Внешняя проверка нулей/шага: [Validation ETA](code-reference.md#validation-eta).

---

Детали модулей и форматов данных: [`repository-layout.md`](../repository-layout.md), [`pipeline-and-algorithms.md`](../pipeline-and-algorithms.md).
