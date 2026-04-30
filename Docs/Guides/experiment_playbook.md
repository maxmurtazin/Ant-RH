# Ant-RH Experiment Playbook

## 0. Setup

Run:

make dashboard-next

Open:
http://localhost:3000

---

## 1. Health Check

Run:
make gemma-health

Or dashboard:
Run → Gemma Health

Check:
- llama-cli ok
- models loaded

---

## 2. Baseline ACO

Run:
make run-v12

Or dashboard:
Run → ACO

Monitor:
- best_loss ↓
- mean_loss ↓

---

## 3. Analyze Results

Run:
make analyze-gemma
make lab-journal

Dashboard:
Status / Reports

---

## 4. TopologicalLM

Run:
make topo-train
make topo-eval

Check:
- reward_mean
- unique_candidate_ratio

---

## 5. PPO Improvement (optional)

Run:
make topo-ppo

Check:
- reward_mean improves
- diversity increases

---

## 6. Physics Diagnostics

Dashboard:
Physics Diagnostics

Check:
- self_adjoint_status = ok
- r_mean → ~0.5 (GUE-like)

---

## 7. Operator Analysis

Run:
make pde
make sensitivity

Dashboard:
Operator Analysis

Check:
- formula exists
- sensitivity > 0

---

## 8. Checkpoint

Dashboard:
Create Checkpoint

or:
make checkpoint

---

## 9. Export

Dashboard:
Export ZIP

---

## 10. Iterate

Loop:
ACO → LM → PPO → PDE → Analysis

---

## Key failure modes

- ACO not learning → adjust beta / reward
- TopologicalLM ≈ random → fix reward conditioning
- low diversity → increase temperature / PPO
- operator insensitive → increase geo_weight
- physics broken → fix self-adjoint

