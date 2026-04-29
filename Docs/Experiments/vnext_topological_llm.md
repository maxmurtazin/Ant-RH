# VNext Topological LLM

## Goal

Train a token model over braid/DTES episodes and use it to generate candidate symbolic structures for downstream scoring.

## What changed

- Added DTES-braid DSL, symbolic tokenizer, tiny transformer model, proxy executor, training, evaluation, and report generation.
- Added generation controls for repetition penalty, no-repeat n-grams, top-p sampling, and candidate deduplication.

## Results

- Latest report: raw valid generation works with `valid_braid_ratio = 0.94`.
- Raw mean reward is `0.9013323550357765`.
- Deduplicated mean reward is `-0.7915124935111636`.
- Random baseline mean reward is `1.1691048539623852`.
- Unique candidate ratio is `0.25`, with `150` duplicates in the reported run.

## Failure modes

- Mode collapse remains strong around a narrow family close to `[4, -2, -1, ...]`.
- Deduplication reduces reward, which indicates the model is repeatedly finding a few high-scoring proxy patterns.
- The executor is still proxy-based, so even improved reward would not yet prove useful operator behavior.

## Conclusion

VNext shows that valid generation is possible, but the current model is not yet diverse enough and does not beat random after deduplication.
# VNext Topological LLM

## What was tried
- A small transformer was trained over a braid/DTES DSL to generate candidate symbolic sequences.

## Results
- Current reporting says the model is not yet better than random.

## Limitations
- The model is heuristic and does not prove RH.
- Evaluation quality depends strongly on the executor reward design and dataset quality.
