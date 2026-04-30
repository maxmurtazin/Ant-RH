SHELL := /bin/bash
.PHONY: install smoke-v12 run-v12 full-v12 artin selberg operator aco rl stability clean-v12 gemma-test gemma-planner-test gemma-analyzer-test aco-gemma analyze-gemma lab-journal paper pde sensitivity topo-dataset topo-train topo-eval topo-report topo-all topo-ppo topo-ppo-report literature docs help install-tts voices help-voice-v2 help-stream help-chat help-chat-memory clear-help-memory study refresh-help gemma-health dashboard-api dashboard-next dashboard-next-no-reload web-install web checkpoint exports
.PHONY: dashboard-light dashboard-next-light

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

lab-journal:
	$(PY) analysis/gemma_lab_journal.py \
		--backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

paper:
	$(PY) analysis/gemma_paper_writer.py \
		--backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

pde:
	$(PY) analysis/operator_pde_discovery.py \
		--operator runs/artin_operator_structured.npy \
		--k 32 \
		--use_llm True \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

sensitivity:
	$(PY) validation/operator_sensitivity_test.py \
		--n_sets 8 \
		--words_per_set 50 \
		--max_length 6 \
		--max_power 4 \
		--n_points 128 \
		--top_k_geodesics 500

topo-dataset:
	$(PY) validation/train_topological_llm.py --build_dataset_only True

topo-train:
	$(PY) validation/train_topological_llm.py \
		--epochs 20 \
		--batch_size 32 \
		--seq_len 128 \
		--out_dir runs/topological_lm

topo-eval:
	$(PY) validation/eval_topological_llm.py \
		--model runs/topological_lm/model.pt \
		--tokenizer runs/topological_lm/tokenizer.json \
		--num_candidates 200

topo-report:
	$(PY) analysis/topological_llm_report.py

topo-ppo:
	$(PY) validation/train_topological_ppo.py \
		--model runs/topological_lm/model.pt \
		--tokenizer runs/topological_lm/tokenizer.json \
		--updates 200 \
		--batch_size 32 \
		--max_new_tokens 32 \
		--lr 1e-5 \
		--out_dir runs/topological_ppo

topo-ppo-report:
	$(PY) analysis/topological_ppo_report.py

topo-all:
	$(MAKE) topo-train
	$(MAKE) topo-eval
	$(MAKE) topo-report

literature:
	$(PY) analysis/gemma_literature_study.py \
		--literature_dir Docs/Literature \
		--backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

docs:
	$(PY) analysis/gemma_docs_builder.py \
		--backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

help:
	$(PY) help/gemma_help_agent.py \
		--question "$(q)" \
		--voice True

install-tts:
	python3 -m pip install pyttsx3

voices:
	$(PY) help/gemma_help_agent.py --list_voices True --tts_backend pyttsx3

help-voice-v2:
	$(PY) help/gemma_help_agent.py \
		--question "$(q)" \
		--voice True \
		--tts_backend pyttsx3 \
		--voice_rate 150 \
		--voice_volume 0.85 \
		--soft_mode True

help-stream:
	$(PY) help/gemma_help_agent.py \
		--question "$(q)" \
		--voice True \
		--tts_backend say \
		--voice_name Samantha \
		--voice_rate 150 \
		--tts_max_chars 180 \
		--stream True \
		--stream_chunk_chars 180 \
		--stream_pause 0.15

help-chat:
	$(PY) help/gemma_help_agent.py \
		--interactive True \
		--voice True \
		--tts_backend say \
		--voice_name Ava \
		--voice_rate 145 \
		--stream True \
		--stream_chunk_chars 180

help-chat-memory:
	$(PY) help/gemma_help_agent.py \
		--interactive True \
		--memory True \
		--voice True \
		--tts_backend say \
		--voice_name Samantha \
		--voice_rate 145 \
		--stream True

clear-help-memory:
	rm -f runs/help_memory.jsonl runs/help_memory_summary.md

study:
	$(PY) analysis/gemma_project_study.py \
		--root . \
		--backend llama_cpp \
		--llama_cli $(LLAMA_CLI) \
		--model_path $(GEMMA_ANALYZER)

refresh-help:
	$(MAKE) study
	$(MAKE) analyze-gemma
	$(MAKE) lab-journal

clean-v12:
	rm -rf runs/artin_* runs/selberg_* runs/operator_* logs/v12

api:
	uvicorn api.server:app --host 127.0.0.1 --port 8080

dashboard:
	python3 scripts/run_dashboard.py --port 8080 --reload True

dashboard-no-reload:
	python3 scripts/run_dashboard.py --port 8080 --reload False

dashboard-8081:
	python3 scripts/run_dashboard.py --port 8081 --reload True

api-install:
	python3 -m pip install fastapi uvicorn pydantic psutil

gemma-health:
	python3 analysis/gemma_health_check.py \
		--llama_cli $(LLAMA_CLI) \
		--planner_model $(GEMMA_PLANNER) \
		--analyzer_model $(GEMMA_ANALYZER) \
		--timeout 60

web-install:
	cd web && npm install
	cd web && npm install recharts html2canvas

web:
	cd web && npm run dev

dashboard-api:
	python3 scripts/run_dashboard.py \
		--port 8084 \
		--reload True \
		--write_port_file runs/dashboard_port.txt

dashboard-light:
	LOW_RESOURCE_MODE=True python3 scripts/run_dashboard.py --port 8084 --reload False

dashboard-next:
	@echo "Starting Ant-RH dashboard..."
	@mkdir -p runs
	@rm -f runs/dashboard_port.txt
	@trap 'kill 0' INT TERM EXIT; \
		python3 scripts/run_dashboard.py \
			--port 8084 \
			--reload True \
			--write_port_file runs/dashboard_port.txt & \
		while [ ! -f runs/dashboard_port.txt ]; do sleep 0.2; done; \
		PORT=$$(cat runs/dashboard_port.txt); \
		echo "Backend: http://127.0.0.1:$$PORT"; \
		echo "Frontend: http://localhost:3000"; \
		cd web && NEXT_PUBLIC_API_BASE=http://127.0.0.1:$$PORT npm run dev

dashboard-next-no-reload:
	@trap 'kill 0' INT TERM EXIT; \
		rm -f runs/dashboard_port.txt; \
		python3 scripts/run_dashboard.py --port 8084 --reload False --write_port_file runs/dashboard_port.txt & \
		while [ ! -f runs/dashboard_port.txt ]; do sleep 0.2; done; \
		PORT=$$(cat runs/dashboard_port.txt); \
		cd web && NEXT_PUBLIC_API_BASE=http://127.0.0.1:$$PORT npm run dev

dashboard-next-light:
	@mkdir -p runs
	@rm -f runs/dashboard_port.txt
	@trap 'kill 0' INT TERM EXIT; \
		LOW_RESOURCE_MODE=True python3 scripts/run_dashboard.py --port 8084 --reload False --write_port_file runs/dashboard_port.txt & \
		while [ ! -f runs/dashboard_port.txt ]; do sleep 0.2; done; \
		PORT=$$(cat runs/dashboard_port.txt); \
		cd web && NEXT_PUBLIC_API_BASE=http://127.0.0.1:$$PORT npm run dev

full-dashboard:
	$(MAKE) dashboard-next

checkpoint:
	curl -X POST http://127.0.0.1:$$(cat runs/dashboard_port.txt)/exports/create \
	  -H "Content-Type: application/json" \
	  -d '{"name":"manual_cli_checkpoint","reason":"manual_cli"}'

exports:
	curl http://127.0.0.1:$$(cat runs/dashboard_port.txt)/exports

screenshots:
	ls -lh runs/screenshots

