# Справочник: функции, методы, роли модулей

Полные таблицы, вынесенные из [`architecture.md`](architecture.md) для краткости обзорной страницы.

## FractalDTESACOZeta

Файл: `fractal_dtes_aco_zeta_metrics.py`.

| Категория | Метод |
|------|--------|
| Жизненный цикл | `__init__`, `run` |
| Сетка / признаки | `evaluate_grid`, `compute_multiscale_features`, `compute_window_features` |
| Дерево | `build_dyadic_tree`, `_make_node`, `_split_node_recursive`, `indices_in_window` |
| Узлы | `aggregate_node_statistics`, `compute_node_energy` |
| Граф | `build_graph`, `add_undirected_edge` |
| Pheromone / ACO | `initialize_pheromones`, `run_aco`, `evaporate_pheromones`, `reinforce_pheromones` |
| Агенты / путь | `make_ant`, `sample_ant_path`, `update_ant_memory`, `compute_agent_value`, `compute_agent_path_bonus`, `agent_type_channel_scale`, `channel_agreement_weights` |
| Межканальные веса | `cross_channel_pheromone`, `mixed_pheromone` |
| Барьер / сходство | `compute_node_stability`, `interval_overlap_ratio`, `compute_barrier`, `tree_distance_proxy`, `signature_distance` |
| Скоринг пути | `evaluate_path`, `path_coherence`, `path_oscillation_penalty` (static) |
| Стат. узлы/окна | `local_roughness` (static), `count_sign_changes` (static) |
| Кандидаты | `rank_candidate_nodes`, `refine_candidates`, `local_refinement`, `verify_zero_candidate`, `merge_close_candidates` |
| Диагностика | `node_pheromone_mass` |
| Модуль (не методы класса) | Внизу файла: `_gini`, `_top_fraction_mass`, `_normalized_entropy`, `_safe_corr`, `_run_lengths`, … |

`fractal_dtes_aco_zeta_crosschannel.py` содержит **параллельную копию** тех же dataclass-ов и класса `FractalDTESACOZeta` (для экспериментов cross-channel); ведите изменения синхронно с `metrics` или вынесите импорт.

## Core module

Файл: `core/fractal_dtes_crosschannel_explore_eta_clean.py`. Класс `Timer`: `__init__`, `log` — тайминги этапов.

| Функция | Назначение |
|--------|------------|
| `hardy_z`, `abs_zeta` | Значения на критической линии (mpmath) |
| `refine_bracket`, `refine_min_abs_zeta` | Локальное уточнение вокруг минимума \|ζ\| |
| `evaluate_grid` | Равномерная сетка по t, `log\|ζ\|`, Z и др. |
| `local_minima_indices`, `sign_change_intervals` | Индексы минимумов и смена знака Z |
| `multiscale_prominence` | Многомасштабный «выступ» вокруг точки |
| `build_candidate_pool` | Пулы + скоринг по каналам, метрики в dict |
| `select_with_exploration` | Бины + exploration-gated отбор |
| `refine_candidates` | Subgrid refine у отобранных |
| `merge_close` | Слияние близких t |
| `make_edge_anchors` | Якоря у границ интервала |
| `save_candidate_json`, `save_edgeaware_json`, `json_dump` | Запись JSON |
| `main` | CLI |

## Hybrid module

Файл: `hybrid/hybrid_dtes_guided_scan.py`.

| Функции / класс | Роль |
|----------------|------|
| `load_dtes_candidates`, `extract_sequence`, `extract_t`, `extract_score` | Чтение пула кандидатов из JSON |
| `merge_intervals`, `build_windows`, `total_window_length` | Окна вокруг кандидатов, слияние |
| `hardy_z`, `abs_zeta`, `bisect_root` | Скан и уточнение нулей в окнах |
| `scan_window` | Плотный проход + корни в одном окне |
| `merge_close_zeros` | Слияние дублей |
| `ETA` | `__init__`, `update` — EMA шага печати ETA по окнам |
| `write_csv`, `main` | Выход и CLI |

## Validation distance

**`validation/distance_analysis.py`.** Основной поток: `load_truth_json`, `load_dtes_json` / `load_distances_csv` → `nearest_distances_from_arrays` → `compute_stats`, `recall_at_windows` → `write_summary` + при опции `plot_*`.

| Функция | Роль |
|---------|------|
| `load_json`, `extract_sequence`, `extract_t` | JSON → массивы t |
| `nearest_distances_from_arrays` | Расстояния до ближайшего «истинного» нуля |
| `compute_stats` | mean, quantiles, recall по порогам |
| `recall_at_windows` | Sweep по гипотетическим окнам |
| `write_csv`, `write_summary` | `distance_*` и markdown summary |
| `plot_histogram`, `plot_cdf`, `plot_window_recall` | Необязательные графики |

**`fractal_dtes_aco_zeta_all_zeros_scan.py` (и копия в `validation/`)** — эталон по Hardy Z: `scan_zeros` → `bisect_root` → `merge_close_zeros` → `write_csv` / `write_txt` / `main`. Вспомогательные: `hardy_z`, `abs_zeta`, `fmt_time`.

## Validation figures

**`validation/figures_from_results.py`.** `build_figures`: вызывает `plot_aco_convergence`, `plot_eta`, `plot_stage_timings`, `plot_hit_rate`, `plot_agent_summary`, `plot_channel_alignment`, `plot_pheromone_metrics`, `plot_ramsey_summary`, при необходимости `copy_existing_pngs`, `make_latex_include`.

## Validation ETA

**`validation/validate_zeros_and_spacing_eta.py`.** `ETAState`, `verify_candidates_eta`, `scan_hardy_sign_changes_eta`, `validate_pipeline_eta`, `main` (требует внешние `validate_zeros_and_spacing`).
