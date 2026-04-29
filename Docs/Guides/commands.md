# Commands

## Core pipeline

- `make run-v12`: run the default Hydra-configured V12 pipeline.
- `make smoke-v12`: run the smoke configuration.
- `make full-v12`: run the full V12 configuration.
- `make artin`: generate symbolic Artin-word data.
- `make selberg`: compute Selberg-style loss from lengths and zeros.
- `make operator`: build the operator.
- `make aco`: run ACO search without Gemma planning.
- `make aco-gemma`: run ACO with Gemma planner assistance.
- `make rl`: run RL training.
- `make stability`: compute operator stability diagnostics.

## Analysis and reports

- `make analyze-gemma`: run the metrics analyzer.
- `make lab-journal`: append a journal entry from current artifacts.
- `make paper`: build the paper draft.
- `make pde`: run PDE-style operator discovery.
- `make sensitivity`: run operator sensitivity diagnostics.
- `make study`: rebuild project memory from code and outputs.
- `make literature`: rebuild literature memory from `Docs/Literature`.
- `make docs`: rebuild documentation files.

## TopologicalLM

- `make topo-dataset`: build the TopologicalLM dataset only.
- `make topo-train`: train the TopologicalLM.
- `make topo-eval`: evaluate raw and deduplicated candidates.
- `make topo-report`: write the TopologicalLM Markdown report.
- `make topo-all`: run train, eval, and report in sequence.

## Help and TTS

- `make help`: answer a single help question with voice.
- `make help-chat`: start the interactive help agent.
- `make help-chat-memory`: start help agent with persistent memory enabled.
- `make help-stream`: stream spoken output with `say`.
- `make help-voice-v2`: use the `pyttsx3` TTS backend.
- `make voices`: list `pyttsx3` voices.
- `make clear-help-memory`: remove stored help memory files.
- `make install-tts`: install `pyttsx3`.

## Utility

- `make gemma-test`: check `llama-cli` and planner model wiring.
- `make gemma-planner-test`: test the Gemma planner wrapper.
- `make gemma-analyzer-test`: test the Gemma analyzer wrapper.
- `make refresh-help`: run `study`, `analyze-gemma`, and `lab-journal`.
- `make clean-v12`: remove run artifacts under `runs/artin_*`, `runs/selberg_*`, `runs/operator_*`, and `logs/v12`.
