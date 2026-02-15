
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import sys
import os
import numpy as np

# Add current directory to path
sys.path.append(os.getcwd())

from BAA.baa_interface import BAAAgent

def visualize_rope_attention():
    print("🚀 Initializing BAA with RoPE for Visualization...")
    
    # Initialize agent with smaller dims for clearer visualization
    # state_dim=32, action_dim=4, embed_dim=64, n_layers=2
    # But BAAAgent defaults are fixed in constructor arguments mostly, 
    # except state/action. We can just use the standard one.
    agent = BAAAgent(
        state_dim=348, 
        action_dim=17
    )
    
    # Create synthetic sequence with a distinct pattern
    # Batch=1, Seq=20
    seq_len = 20
    states = torch.randn(1, seq_len, 348)
    rtg = torch.randn(1, seq_len, 1)
    
    print(f"🎨 Generating Attention Maps for sequence length {seq_len}...")
    
    with torch.no_grad():
        output = agent.model(states, rewards_to_go=rtg)
        attn_maps = output["attention_maps"] # List of [batch, heads, seq, seq]
        
    # We will visualize Layer 0, Head 0
    # RoPE should show diagonal banding or relative decay
    
    layer_idx = 0
    head_idx = 0
    
    if len(attn_maps) <= layer_idx:
        print("❌ Model has fewer layers than expected.")
        return

    # Extract map: [batch, heads, seq, seq] -> [seq, seq]
    attn_matrix = attn_maps[layer_idx][0, head_idx].numpy()
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(attn_matrix, cmap="viridis", annot=False)
    plt.title(f"RoPE Attention Pattern (Layer {layer_idx}, Head {head_idx})\nNote the relative positional structure!", fontsize=14)
    plt.xlabel("Key Position (Source)")
    plt.ylabel("Query Position (Target)")
    
    save_path = "rope_attention_pattern.png"
    plt.savefig(save_path)
    print(f"✅ Visualization saved to {save_path}")
    
    # Also visualize a deeper layer to see if it changes
    if len(attn_maps) > 1:
        layer_idx = 1
        attn_matrix = attn_maps[layer_idx][0, head_idx].numpy()
        plt.figure(figsize=(10, 8))
        sns.heatmap(attn_matrix, cmap="magma", annot=False)
        plt.title(f"RoPE Attention Pattern (Layer {layer_idx}, Head {head_idx})", fontsize=14)
        plt.xlabel("Key Position")
        plt.ylabel("Query Position")
        plt.savefig("rope_attention_pattern_layer1.png")
        print(f"✅ Second layer visualization saved to rope_attention_pattern_layer1.png")

if __name__ == "__main__":
    try:
        visualize_rope_attention()
    except Exception as e:
        print(f"❌ Error during visualization: {e}")
