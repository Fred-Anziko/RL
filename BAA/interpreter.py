import streamlit as st
import graphviz
import torch
import numpy as np
import os
from .orchestrator import AgenticDTT

# --- 1. Live Data Connection ---
def get_decision_trace(model, states, rtg, layer_idx=0):
    """
    Retrieves real-time routing probabilities from the AgenticDTT model.
    Inputs:
        - model: The AgenticDTT model instance
        - states: Input state tensor [batch, seq_len, state_dim]
        - rtg: Reward-to-Go tensor [batch, seq_len, 1]
        - layer_idx: Index of the tree layer to extract
    Outputs:
        - formatted_trace: Dictionary with node routing probabilities for the specified layer
    """
    # model.get_logic_path returns [{layer_0: {node_0: ...}}, ...]
    full_trace = model.get_logic_path(states, None, rtg)
    
    if layer_idx >= len(full_trace):
        return None
        
    layer_key = f"layer_{layer_idx}"
    layer_data = full_trace[layer_idx][layer_key]
    
    formatted_trace = {}
    for node_name, node_probs in layer_data.items():
        # Mean probabilities over the batch/sequence for visualization
        p_right = float(np.mean(node_probs["go_right_prob"]))
        
        formatted_trace[node_name] = {
            "go_right_prob": p_right,
            "go_left_prob": 1.0 - p_right,
            "is_frozen": False 
        }
    return formatted_trace

# --- 2. Visualization Engine ---
def render_decision_tree(trace, depth=3):
    if trace is None:
        return None
        
    dot = graphviz.Digraph(comment='Agentic Logic Tree')
    dot.attr(rankdir='TB')
    
    num_nodes = 2**depth - 1
    for i in range(num_nodes):
        node_key = f"node_{i}"
        if node_key not in trace:
            continue
            
        node_data = trace[node_key]
        p_right = node_data['go_right_prob']
        
        # Color Logic: Red (Left) -> Grey -> Green (Right)
        green_intensity = int(255 * p_right)
        red_intensity = int(255 * (1 - p_right))
        hex_col = f'#{red_intensity:02x}{green_intensity:02x}88'
        
        label = f"Node {i}\nP(Right): {p_right:.2f}"
        dot.node(str(i), label, shape='box', style='filled', fillcolor=hex_col, fontcolor='white')
        
        left_child = 2 * i + 1
        right_child = 2 * i + 2
        
        if left_child < 2**depth - 1:
            width_l = str(1 + 4 * (1 - p_right))
            width_r = str(1 + 4 * p_right)
            dot.edge(str(i), str(left_child), label="L", penwidth=width_l)
            dot.edge(str(i), str(right_child), label="R", penwidth=width_r)
            
    return dot

def load_live_model(checkpoint_path="humanoid_model.pt"):
    # Default dimensions (can be overridden by checkpoint)
    state_dim = 348 
    action_dim = 17
    
    if os.path.exists(checkpoint_path):
        try:
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            state_dict = checkpoint['agentic_dtt'] if isinstance(checkpoint, dict) and 'agentic_dtt' in checkpoint else checkpoint
            
            # Dynamic Dimension Detection: Look at embed_state.weight
            if 'embed_state.weight' in state_dict:
                state_dim = state_dict['embed_state.weight'].shape[1]
            if 'policy_head.weight' in state_dict:
                action_dim = state_dict['policy_head.weight'].shape[0]
                
            model = AgenticDTT(state_dim=state_dim, action_dim=action_dim, embed_dim=256, n_layers=6)
            model.load_state_dict(state_dict)
            print(f"✓ Loaded weights from {checkpoint_path} (Detected State Dim: {state_dim})")
            return model, state_dim
        except Exception as e:
            print(f"Failed to load weights: {e}")
    
    # Fallback to random weights if no checkpoint found or load failed
    model = AgenticDTT(state_dim=state_dim, action_dim=action_dim, embed_dim=256, n_layers=6)
    return model, state_dim

def run_ui():
    st.set_page_config(layout="wide", page_title="BAA Neural Debugger")
    st.title("🧠 Neural Debugger: Agentic DTT")

    model, state_dim = load_live_model()

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("🎛️ Control Center")
        target_rtg = st.slider("Target Reward-to-Go (Signal Confidence)", 0.0, 10.0, 1.0)
        selected_layer = st.number_input("Visualized Layer Index", min_value=0, max_value=5, value=0)
        
        st.divider()
        st.subheader("🚫 Surgical Pruning")
        node_to_prune = st.number_input("Tree Node ID", min_value=0, max_value=127, step=1)
        direction = st.radio("Enforce Branch", ["Left", "Right"])
        
        if st.button("Apply Pruning & Lock Weights"):
            success = model.freeze_node(selected_layer, node_to_prune, direction.lower())
            if success:
                st.success(f"Node {node_to_prune} successfully locked to {direction}.")
            else:
                st.error("Pruning failed.")

    with col2:
        st.header("🌳 Active Decision Backbone")
        # Dynamically generated dummy state
        dummy_state = torch.zeros(1, 1, state_dim)
        dummy_rtg = torch.tensor([[[target_rtg]]])
        
        active_trace = get_decision_trace(model, dummy_state, dummy_rtg, layer_idx=selected_layer)
        graph = render_decision_tree(active_trace)
        
        if graph:
            st.graphviz_chart(graph, use_container_width=True)
        
        if active_trace:
            probs = np.array([v['go_right_prob'] for v in active_trace.values()])
            entropy = -np.mean(probs * np.log(probs + 1e-9) + (1-probs) * np.log(1-probs + 1e-9))
            st.metric("Tree Confusion (Entropy)", f"{entropy:.4f}")

if __name__ == "__main__":
    run_ui()