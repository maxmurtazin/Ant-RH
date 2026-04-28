SHELL := /bin/bash
.PHONY: install smoke-v12 run-v12 full-v12 artin selberg operator aco rl stability clean-v12 gemma-test gemma-planner-test gemma-analyzer-test aco-gemma analyze-gemma

PY := python3
PIP := pip3

LLAMA_CLI ?= llama-cli
GEMMA_DIR ?= /Users/machome/models/gemma
GEMMA_PLANNER ?= $(GEMMA_DIR)/gemma-3-1b-it-Q4_K_M.gguf
GEMMA_ANALYZER ?= $(GEMMA_DIR)/gemma-3-1b-it-Q5_K_M.gguf

check-llama:
	@which llama-cli >/dev/null || (echo "❌ llama-cli not found. Run: brew install llama.cpp" && exit 1)

install:
	$(PIP) install -U hydra-core omegaconf tqdm numpy scipy matplotlib torch

smoke-v12:
	$(PY) scripts/run_v12_hydra.py --config-name v12_smoke

run-v12:
	$(PY) scripts/run_v12_hydra.py --config-name config

full-v12:
	$(PY) scripts/run_v12_hydra.py --config-name v12_full

artin:
	$(PY) -m core.artin_symbolic_billiard --num_samples 10000 --max_length 8 --max_power 6 --out_dir runs/

selberg:
	$(PY) -m validation.selberg_trace_loss --lengths runs/artin_lengths.csv --zeros data/zeta_zeros.txt --sigma 0.5 --m_max 5 --out_dir runs/

operator:
	$(PY) -m core.artin_operator --n_points 256 --sigma 0.3 --top_k_geodesics 500 --geodesics runs/artin_words.json --zeros data/zeta_zeros.txt --out_dir runs/

aco:
	$(PY) -m core.artin_aco --num_ants 64 --num_iters 50 --max_length 8 --max_power 6 --alpha 1.0 --beta 2.0 --rho 0.1 --zeros data/zeta_zeros.txt --out_dir runs/

rl:
	$(PY) -m validation.artin_rl_train --num_updates 50 --steps_per_update 1024 --max_length 8 --max_power 6 --lr 3e-4 --pheromone_bias 0.5 --eval_operator_every 20 --target_zeros_path data/zeta_zeros.txt --out_dir runs/artin_rl

stability:
	$(PY) -m validation.operator_stability_report --operator runs/artin_operator.npy --zeros data/zeta_zeros.txt --k 128 --out runs/operator_stability_report.json

gemma-test:
	$(MAKE) check-llama
	$(LLAMA_CLI) -m $(GEMMA_PLANNER) -p "Return JSON only: [[1,2,-1]]" -n 64

gemma-planner-test:
	$(PY) -m core.gemma_planner --test --llama_cli $(LLAMA_CLI) --model_path $(GEMMA_PLANNER)

gemma-analyzer-test:
	$(PY) analysis/gemma_analyzer.py --test --llama_cli $(LLAMA_CLI) --model_path $(GEMMA_ANALYZER)

aco-gemma:
	$(MAKE) check-llama
	$(PY) -m core.artin_aco \
		--num_ants 32 \
		--num_iters 20 \
		--max_length 8 \
		--max_power 6 \
		--use_planner True \
		--planner_backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--planner_model $(GEMMA_PLANNER)

analyze-gemma:
	$(MAKE) check-llama
	$(PY) analysis/gemma_analyzer.py \
		--backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

clean-v12:
	rm -rf runs/artin_* runs/selberg_* runs/operator_* logs/v12

