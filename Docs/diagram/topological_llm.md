# TopologicalLM — детальная диаграмма

Этот файл выделяет только ветку TopologicalLM внутри Ant-RH: от построения датасета до отчёта по raw/dedup метрикам.

## 1. Полный контур TopologicalLM

```mermaid
flowchart TD
  ACOH["runs/artin_aco_history.csv"]
  ACOB["runs/artin_aco_best.json"]
  JOUR["runs/lab_journal.jsonl\n(if present)"]
  SYN["synthetic episodes\nvalidation/train_topological_llm.py"]

  DSL["core/dtes_braid_dsl.py\nserialize_episode / serialize_from_aco_logs"]
  TOK["core/braid_tokenizer.py"]
  TRAIN["validation/train_topological_llm.py"]
  MODEL["core/topological_llm.py\nTopologicalTinyLM"]

  GEN["generate()\nno_repeat_ngram\nrepetition_penalty\ntop-k / top-p / temperature"]
  DECODE["token decode -> braid_tokens_to_artin_word"]
  DEDUP["dedup by tuple(word)"]
  EXEC["core/dtes_braid_executor.py"]
  EVAL["validation/eval_topological_llm.py"]
  REPORT["analysis/topological_llm_report.py"]

  ACOH --> DSL
  ACOB --> DSL
  JOUR --> DSL
  SYN --> DSL
  DSL --> TOK
  TOK --> TRAIN
  TRAIN --> MODEL
  MODEL --> GEN
  GEN --> DECODE
  DECODE --> DEDUP
  DEDUP --> EXEC
  EXEC --> EVAL
  EVAL --> REPORT
```

## 2. Контур обучения

```mermaid
flowchart LR
  LOGS["ACO logs"]
  JOURNAL["Journal-derived records"]
  SYNTH["Synthetic diverse records"]
  RECORDS["build_dataset_records()"]
  DSL["DSL text episodes"]
  TOK["Tokenizer fit / encode"]
  BATCH["TokenDataset + DataLoader"]
  MODEL["TopologicalTinyLM"]
  LOSS["Cross-entropy"]
  CKPT["model.pt\ntrain_history.csv\nconfig.json"]

  LOGS --> RECORDS
  JOURNAL --> RECORDS
  SYNTH --> RECORDS
  RECORDS --> DSL
  DSL --> TOK
  TOK --> BATCH
  BATCH --> MODEL
  MODEL --> LOSS
  LOSS --> CKPT
```

## 3. Контур генерации и оценки

```mermaid
flowchart LR
  CTX["context_ids = BOS + <episode>"]
  MODEL["TopologicalTinyLM"]
  SAMPLE["sampling\n- temperature\n- top_k\n- top_p\n- no_repeat_ngram_size\n- repetition_penalty"]
  TOKENS["generated token ids"]
  WORD["decoded Artin word"]
  DUP["exact duplicate check"]
  EXEC["evaluate_braid_candidate()"]
  METRICS["raw / dedup metrics\nunique_candidate_ratio\nduplicate_count\nattempts_used"]

  CTX --> MODEL
  MODEL --> SAMPLE
  SAMPLE --> TOKENS
  TOKENS --> WORD
  WORD --> DUP
  DUP --> EXEC
  EXEC --> METRICS
```

## 4. Внутри executor

```mermaid
flowchart TD
  WORD["Artin word"]
  VALID["validity check\nnonzero / length / hyperbolic"]
  LENGTH["length_score"]
  SPEC["spectral_score"]
  STAB["stability_score"]
  DIV["diversity_score\n+ repeat penalty"]
  SUM["weighted reward"]
  CLIP["clip to [-10, 10]"]

  WORD --> VALID
  VALID --> LENGTH
  VALID --> SPEC
  VALID --> STAB
  VALID --> DIV
  LENGTH --> SUM
  SPEC --> SUM
  STAB --> SUM
  DIV --> SUM
  VALID --> SUM
  SUM --> CLIP
```

## 5. Raw vs dedup отчётность

```mermaid
flowchart TB
  CANDS["generated candidates"]
  RAW["raw metrics\nmean / median / std\nvalid ratio"]
  DEDUP["deduplicated set"]
  DEDUPM["dedup metrics\nmean / median / std\nvalid ratio"]
  DIAG["diagnosis"]

  CANDS --> RAW
  CANDS --> DEDUP
  DEDUP --> DEDUPM
  RAW --> DIAG
  DEDUPM --> DIAG
```

## 6. Текущие проблемные места

```mermaid
flowchart TD
  SYN["Synthetic data bias"]
  GEN["Sampler collapse"]
  DUP["Many duplicate words"]
  EXEC["Proxy reward"]
  OUT["Dedup reward < random"]

  SYN --> GEN
  GEN --> DUP
  DUP --> OUT
  EXEC --> OUT
```

## 7. Текущее состояние по последнему отчёту

- Raw valid generation работает.
- `unique_candidate_ratio = 0.25`.
- `duplicate_count = 150`.
- Raw mean reward положительный, но dedup mean reward отрицательный.
- В dedup-сравнении модель хуже random baseline.
- Основная проблема сейчас: mode collapse вокруг узкого семейства слов.
