# Ant-RH — актуальные диаграммы

Ниже собраны актуальные схемы по текущему состоянию репозитория: основной пайплайн, расширение через TopologicalLM, поток данных и место Gemma-агентов. Диаграммы отражают реальное состояние системы, включая слабые места: plateau в ACO, низкую чувствительность оператора и mode collapse в TopologicalLM.

## 1. Архитектура репозитория

```mermaid
flowchart TB
  subgraph CORE["core/"]
    ARTIN["artin_symbolic_billiard.py"]
    OP["artin_operator.py / artin_operator_structured.py"]
    ACO["artin_aco.py"]
    RLENV["artin_rl_env.py"]
    RLPOL["artin_rl_policy.py"]
    STAB["spectral_stabilization.py"]
    DSL["dtes_braid_dsl.py"]
    TOK["braid_tokenizer.py"]
    TLM["topological_llm.py"]
    EXEC["dtes_braid_executor.py"]
    LLMRUN["llm_runner.py"]
    GPLAN["gemma_planner.py"]
  end

  subgraph VALID["validation/"]
    SEL["selberg_trace_loss.py"]
    RL["artin_rl_train.py"]
    OREP["operator_stability_report.py"]
    SENS["operator_sensitivity_test.py"]
    TTRAIN["train_topological_llm.py"]
    TEVAL["eval_topological_llm.py"]
  end

  subgraph ANALYSIS["analysis/"]
    GAN["gemma_analyzer.py"]
    JOUR["gemma_lab_journal.py"]
    PSTUDY["gemma_project_study.py"]
    LSTUDY["gemma_literature_study.py"]
    PDE["operator_pde_discovery.py"]
    TREPORT["topological_llm_report.py"]
    DOCS["gemma_docs_builder.py"]
  end

  subgraph HELP["help/"]
    HAGENT["gemma_help_agent.py"]
    TTS["local_tts.py / local_tts_pyttsx3.py"]
  end

  subgraph RUNS["runs/"]
    RART["artin_*.csv / json"]
    RRL["artin_rl/"]
    ROP["operator_*.json / md"]
    RTLM["topological_lm/"]
    RMEM["project/literature/help memory"]
  end

  ARTIN --> SEL
  ARTIN --> OP
  OP --> ACO
  OP --> RL
  OP --> OREP
  STAB --> OREP
  ACO --> RART
  RL --> RRL
  OREP --> ROP
  DSL --> TOK
  TOK --> TLM
  TLM --> EXEC
  EXEC --> TEVAL
  TTRAIN --> RTLM
  TEVAL --> RTLM
  GAN --> RMEM
  JOUR --> RMEM
  PSTUDY --> RMEM
  LSTUDY --> RMEM
  PDE --> ROP
  TREPORT --> RTLM
  DOCS --> RMEM
  LLMRUN --> GPLAN
  LLMRUN --> GAN
  LLMRUN --> HAGENT
  RMEM --> HAGENT
  HAGENT --> TTS
```

## 2. Основной пайплайн Ant-RH

```mermaid
flowchart LR
  A["Artin words\ncore/artin_symbolic_billiard.py"]
  S["Selberg-style loss\nvalidation/selberg_trace_loss.py"]
  O["Operator construction\ncore/artin_operator.py"]
  C["ACO search\ncore/artin_aco.py"]
  R["RL training\nvalidation/artin_rl_train.py"]
  E["Spectral / stability eval\nvalidation/operator_stability_report.py"]
  L["Logging to runs/\nCSV + JSON + MD"]
  G["Gemma agents\nanalyzer / journal / study / help"]

  A --> S
  A --> O
  S --> C
  O --> C
  O --> R
  C --> E
  R --> E
  E --> L
  L --> G
```

## 3. Расширенный пайплайн TopologicalLM

```mermaid
flowchart LR
  LOGS["ACO logs / journal\nruns/artin_aco_history.csv\nruns/artin_aco_best.json"]
  DSL["DTES-Braid DSL\ncore/dtes_braid_dsl.py"]
  TOK["Tokenizer\ncore/braid_tokenizer.py"]
  TRAIN["Tiny Transformer\ncore/topological_llm.py\nvalidation/train_topological_llm.py"]
  GEN["Candidate generation\nvalidation/eval_topological_llm.py"]
  EXEC["DTES executor\ncore/dtes_braid_executor.py"]
  REFINE["ACO refinement placeholder"]
  REP["Topological report\nanalysis/topological_llm_report.py"]

  LOGS --> DSL
  DSL --> TOK
  TOK --> TRAIN
  TRAIN --> GEN
  GEN --> EXEC
  EXEC --> REFINE
  EXEC --> REP
```

## 4. Поток данных

```mermaid
flowchart TD
  H["runs/artin_aco_history.csv"]
  B["runs/artin_aco_best.json"]
  J["runs/lab_journal.jsonl\n(if present)"]
  D["Serialized DSL episodes"]
  T["Token ids"]
  M["TopologicalTinyLM"]
  C["Decoded token sequences"]
  W["Artin words"]
  X["Executor metrics\nvalidity / length / spectral / stability / diversity"]
  R["reward + dedup stats"]
  O["eval_report.json\nbaseline_comparison.csv\nreport.md"]

  H --> D
  B --> D
  J --> D
  D --> T
  T --> M
  M --> C
  C --> W
  W --> X
  X --> R
  R --> O
```

## 5. Контур Gemma-агентов

```mermaid
flowchart TB
  RUNS["runs/*.csv / *.json / *.md"]
  ANALYZER["gemma_analyzer.py"]
  JOURNAL["gemma_lab_journal.py"]
  PROJECT["gemma_project_study.py"]
  LIT["gemma_literature_study.py"]
  DOCS["gemma_docs_builder.py"]
  HELP["gemma_help_agent.py"]
  MEM["project_memory.md\nliterature_memory.md\nhelp_memory.jsonl"]

  RUNS --> ANALYZER
  RUNS --> JOURNAL
  RUNS --> PROJECT
  RUNS --> DOCS
  RUNS --> HELP
  ANALYZER --> MEM
  JOURNAL --> MEM
  PROJECT --> MEM
  LIT --> MEM
  MEM --> HELP
  MEM --> DOCS
```

## 6. Диаграмма проблемных мест

```mermaid
flowchart TD
  ACO["ACO"]
  OP["Operator sensitivity"]
  TLM["TopologicalLM"]
  EXEC["Executor reward"]
  DOC["Reports / Help"]

  ACO -->|"plateau / unstable learning"| OP
  OP -->|"word changes barely affect operator"| TLM
  TLM -->|"mode collapse\nunique_candidate_ratio = 0.25"| EXEC
  EXEC -->|"reward bounded but proxy-based"| DOC
```

## 7. Ключевые наблюдения

- ACO сейчас не показывает устойчивого улучшения.
- Оператор численно стабилен, но слабо реагирует на изменение Artin words.
- TopologicalLM умеет генерировать валидные кандидаты, но diversity низкая.
- Исполнитель TopologicalLM стал стабильнее по reward, но это всё ещё proxy-оценка.
