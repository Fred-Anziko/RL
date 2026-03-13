# Branching-Attention Agent (BAA)

A Decision Tree Transformer Framework for Agentic Modeling.

## Project Overview

The BAA is a unified ML framework combining Transformers with Differentiable Soft Decision Trees to create an explainable agentic model capable of multi-mode learning (RL, UL, SSL, SL).

## Architecture

- **Frontend (Streamlit)**: Neural Debugger UI at port 5000 — `BAA/interpreter.py`
- **Backend (FastAPI)**: Inference and online learning API at port 8000 — `BAA/app.py`

### Key Modules (`BAA/`)
- `orchestrator.py` — AgenticDTT model, ReplayBuffer
- `brain.py` — AgenticBrain (action selection, training loop)
- `baa_interface.py` — BAAAgent (high-level interface with thread safety)
- `interpreter.py` — Streamlit Neural Debugger UI
- `app.py` — FastAPI inference server
- `ffn.py` — TreeTransformerBlock
- `ddt.py` — Differentiable Decision Tree
- `bayesian.py` — Bayesian linear layers
- `curiosity.py` — CuriosityEngine
- `hindsight.py` — HindsightRelabeler
- `loss.py` — DTTLossEngine
- `rope.py` — Rotary Positional Embeddings

### Model Files
- `humanoid_model.pt` — Main model weights (state_dim=376, action_dim=17)
- `humanoid_model_best.pt` — Best-performing checkpoint

## Workflows

- **Start application**: `streamlit run BAA/interpreter.py --server.port 5000 --server.address 0.0.0.0 --server.headless true --server.enableCORS false --server.enableXsrfProtection false`
- **Backend API**: `uvicorn BAA.app:app --host localhost --port 8000`

## Usage

The FastAPI backend exposes a `/predict` endpoint:
```bash
python3 -c "import requests; r = requests.post('http://localhost:8000/predict', json={'state': [0.1]*376, 'rtg': 10.0, 'session_id': 'test_session'}); print(r.json())"
```

## Recent Improvements

### 1. Real Hindsight Experience Replay (`BAA/hindsight.py`)
`HindsightRelabeler.relabel_episode()` now returns a **list of two samples** per episode:
- **Original sample**: trajectory with the agent's intended RTG goal.
- **Hindsight sample**: the same trajectory relabeled so the RTG at every step is rescaled as if the agent always intended to reach the actual final outcome (`override_total`). This turns failed episodes into valid "virtual successes" for a different target, which is the core HER re-goal operation.
`orchestrator.on_episode_finish()` pushes both samples to the replay buffer.

### 2. Learned Loss Balancing (`BAA/loss.py`)
Replaced all hand-tuned static weights with **uncertainty weighting** (Kendall et al. 2018). `DTTLossEngine` now has 6 learnable `log_sigma` parameters (one per loss component). Effective weight for each loss is `1 / (2 * σ²)` plus a regularizer `log(σ)` that prevents the model from zeroing out any component. The optimizer in `AgenticBrain` covers both model weights and the loss-balancing parameters.

### 3. Decoupled Inference and Training Threads (`BAA/baa_interface.py`)
- **`_episode_queue`**: `record_experience()` posts completed episodes here and returns immediately — no blocking during training.
- **`_training_loop`**: daemon thread that drains the queue, calls `on_episode_finish` and `train_on_buffer`, holding `_weight_lock` only during that window.
- **`_weight_lock`**: separates weight reads (inference) from weight writes (optimizer step).
- **`_traj_lock`**: lightweight lock for the trajectories dict only, independent of weight access.
- Added `get_training_status()` for monitoring queue depth, episode count, and recent metrics.

## Dependencies

All managed via `requirements.txt` (Python 3.12). Key packages:
- `torch==2.2.2` — Neural network engine
- `streamlit==1.53.1` — Frontend UI
- `fastapi` + `uvicorn` — Backend API
- `gymnasium` — RL environment support
- `graphviz` — Decision tree visualization
