# Branching-Attention Agent (BAA)

A Decision Tree Transformer Framework for Agentic Modeling.

## Project Overview

The BAA is a unified ML framework combining Transformers with Differentiable Soft Decision Trees to create an explainable agentic model capable of multi-mode learning (RL, UL, SSL, SL).

## Architecture

- **Frontend (Streamlit)**: Neural Debugger UI at port 5000 — `BAA/interpreter.py`
- **Backend (FastAPI)**: Inference and online learning API at port 8000 — `BAA/app.py`

### Key Modules (`BAA/`)
- `orchestrator.py` — AgenticDTT model, PrioritizedSequenceReplayBuffer, RoutingTracker
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

### 1. RTG Clamping in Hindsight Relabeling (`BAA/hindsight.py`)
`compute_rtg()` now clamps all RTG values to `[-100, 100]` after the hindsight rescaling step. This prevents near-zero episode totals from producing enormous scale factors that would destabilize cross-attention layers during training.

### 2. SSL Loss Wired (`BAA/brain.py`)
`train_on_buffer()` now samples a second "unlabeled" batch from the replay buffer and runs a `no_grad` forward pass to collect its routing trace. This trace is passed as `unlabeled_routing_trace` to the `DTTLossEngine`, activating the path-consistency SSL loss (component 3) for the first time. Previously this code path was architecturally present but never triggered.

### 3. Training Thread Watchdog (`BAA/baa_interface.py`)
- `_last_heartbeat`: timestamp updated every iteration by the training thread.
- `_check_watchdog()`: called from `get_action()` on every inference step. If the training thread has been silent for > 60 s or has died, it is automatically restarted via `_restart_training_thread()`.
- `get_training_status()` now exposes `thread_alive` and `last_heartbeat_age_s` for monitoring.

### 4. Multi-Step Sequence Replay (`BAA/orchestrator.py`)
Replaced `ReplayBuffer` with `PrioritizedSequenceReplayBuffer`. Instead of storing individual timestep transitions, the buffer now creates **overlapping windows of length 8** from each episode. Each window is stored as `[1, 8, dim]`. This gives the Transformer and RoPE attention real temporal context during training (previously they only saw sequences at inference time).

### 5. Prioritized Experience Replay (`BAA/orchestrator.py`)
`PrioritizedSequenceReplayBuffer.sample()` returns `(samples, indices)` using **proportional priority sampling** (`P(i) ∝ priority_i^0.6`). New transitions receive maximum priority. After each training step, `brain.train_on_buffer()` computes per-sample action losses and calls `update_priorities(indices, losses)` so surprising transitions are sampled more frequently.

### 6. Automated Pruning (`BAA/orchestrator.py`, `BAA/brain.py`)
`RoutingTracker` maintains a running mean of `P(Right)` for every `(layer, node)` pair. After 100 observations, nodes whose mean stays above 0.95 or below 0.05 are automatically frozen via `AgenticDTT.freeze_node()`. The tracker is updated from `train_on_buffer()` after every training step and pruning happens in-line via `check_auto_prune()`.

### 7. Benchmark Script (`BAA/benchmark.py`)
`python3 BAA/benchmark.py [--episodes 100] [--seed 42]` runs BAA and a random-policy baseline head-to-head on Pendulum-v1 (continuous action, no MuJoCo license required). Prints per-episode rewards, a summary table, and a learning trend (second-half vs first-half mean) to quantify whether BAA improves during the run.

### Prior Improvements
- **Real Hindsight Experience Replay** (`hindsight.py`): two samples per episode (original + hindsight re-goal).
- **Learned Loss Balancing** (`loss.py`): 6 learnable `log_sigma` parameters via Kendall et al. 2018.
- **Decoupled Inference/Training Threads** (`baa_interface.py`): episode queue, `_weight_lock`, `_traj_lock`.

## Dependencies

All managed via `requirements.txt` (Python 3.12). Key packages:
- `torch==2.2.2` — Neural network engine
- `streamlit==1.53.1` — Frontend UI
- `fastapi` + `uvicorn` — Backend API
- `gymnasium` — RL environment support
- `graphviz` — Decision tree visualization
