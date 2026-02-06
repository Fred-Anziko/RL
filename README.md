📑 Project: The Branching-Attention Agent (BAA)
A Decision Tree Transformer Framework for Agentic Modeling

🚀 Overview
The Branching-Attention Agent is a unified framework designed to bring explainability to the sequence-modeling power of Transformers. By replacing the standard MLP layers in a Transformer with Differentiable Soft Decision Trees, we create an agent capable of multi-mode learning (RL, UL, SSL, SL) while maintaining a human-readable logic trace.

🧠 Key Innovations
- **Tree-Transformer Hybrid Block**: Uses Soft Decision Trees as the core reasoning unit.
- **Explicit Bayesian Reasoning**: Implements Bayesian Linear layers in decision routers for robust epistemic uncertainty estimation.
- **Curiosity-Driven Online Learning**: Combines binary entropy and Bayesian KL divergence to drive exploration in unknown or novel state-spaces.
- **Hindsight Reward Re-labeling**: Dynamically re-aligns "failed" trajectories to ground the agent's understanding of reality.
- **Rotary Positional Embeddings (RoPE)**: Leverages Euler's formula to inject relative positional information into the attention mechanism, enabling robust generalization to infinite-horizon tasks.
- **Neural Debugger**: A real-time Streamlit dashboard for manual "logic pruning" and path visualization.

🛠 Technical Architecture

1. The Core Engine
The architecture processes state-action-reward sequences. Within each layer, attention identifies where to look, while the Decision Tree identifies how to categorize the logic.

2. Learning Paradigms
| Mode | Objective | Mechanism |
| :--- | :--- | :--- |
| RL | Maximize Return | Hindsight-aligned Policy Gradient |
| UL | World Modeling | Masked State Reconstruction |
| SSL | Label Propagation | Path-consistency between labeled/unlabeled branches |
| SL | Prediction | Standard MSE/Cross-Entropy on leaf outputs |

🚢 Deployment Strategy
The system is designed as a containerized microservice architecture:
- **Agent Brain (FastAPI)**: High-performance inference and online learning engine.
- **Neural Debugger (Streamlit)**: Visual interface for real-time monitoring and "surgical" pruning of logic branches.

🔧 Neural Debugger Commands
- **Trace Path**: Visualize the high-probability route through the Transformer's layers.
- **Entropy Monitor**: Identify "confusion" spikes where the agent is uncertain.
- **Hard-Pruning**: Force a decision node to a specific branch to enforce safety constraints without re-training.

🧠 Agentic Philosophy
The BAA is defined as **Agentic** because it is **Self-Regulating**. It does not simply follow a fixed policy; it adapts its learning objective based on the quality of environmental signals:
- **No Rewards?** It defaults to **UL** (Unsupervised Learning) to build its own model of the world's physics.
- **Sparse Data?** It uses **SSL** (Semi-Supervised Learning) to infer logic from path-consistency and context.
- **Uncertainty?** It triggers its **Curiosity Engine** to explore unknown logic branches until it gains certainty.

It is a "full-stack" neural brain designed to be dropped into an unknown environment and left to figure things out autonomously.

🛠 Tools
- **Neural Debugger:** A Streamlit-based dashboard for visualizing logic paths and "surgical pruning" of decision nodes.
- **Inference API:** A FastAPI service for serving the agent with background online learning.
'''
python3 -c "import requests; r = requests.post('http://localhost:8000/predict', json={'state': [0.1]*376, 'rtg': 10.0, 'session_id': 'test_session'}); print(r.json())"
'''

