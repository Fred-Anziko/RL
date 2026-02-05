import gymnasium as gym
import torch
import numpy as np
import os
import signal
import sys
from BAA.baa_interface import BAAAgent
from BAA.interpreter import get_decision_trace, render_decision_tree

# Graceful exit handler
def signal_handler(sig, frame):
    print("\n⏳ Interrupt received! Saving weights before exiting...")
    if 'agent' in globals():
        agent.save_weights()
        print("✅ Weights saved successfully.")
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

# 1. Setup Environment (Headless & Infinite Life Mode)
# terminate_when_unhealthy=False: prevents the env from stopping when the agent falls
# 1. Setup Environment (Headless & Infinite Life Mode)
# Using Humanoid-v4 which is standard on Kaggle/Gymnasium
print("🚀 Setting up environment (Headless Mode)...")
try:
    # v4 usually dictates termination on unhealthy. We will handle loop reset manually.
    env = gym.make("Humanoid-v4", render_mode=None, terminate_when_unhealthy=False)
except Exception as e:
    print(f"⚠️ Failed to create Humanoid-v4: {e}")
    # Fallback to v2/v3 if needed, or just raise
    try:
        env = gym.make("Humanoid-v3", render_mode=None)
    except:
        raise e

# 2. Initialize the Unified Agent
# Ensure we map to the correct device automatically
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device: {device}")

agent = BAAAgent(
    state_dim=376, 
    action_dim=17, 
    weight_path="humanoid_model.pt" # Saves to current working dir (/kaggle/working)
)

# Initialize
obs, info = env.reset()
rtg = 15000.0
total_reward = 0
current_segment_reward = 0

print("🚀 Starting Infinite Life Training on Kaggle...")
print(f"💾 Model will be saved to: {os.path.abspath('humanoid_model.pt')}")

try:
    for step in range(5000000): # Increased horizon for long runs
        # 1. Get Action
        action_tensor, trace, curiosity = agent.get_action(obs, rtg)
        action = action_tensor.detach().cpu().numpy().flatten()
        
        # 2. Step the environment
        next_obs, reward, terminated, truncated, info = env.step(action)
        
        # Auto-reset for "Infinite Life" feel
        # if terminated or truncated:
        #     next_obs, info = env.reset()

        
        current_segment_reward += reward
        total_reward += reward
        
        # 3. Continuous Learning Logic
        is_learning_trigger = (step % 1000 == 0 and step > 0)
        
        agent.record_experience(
            session_id="infinite_life_kaggle",
            state=torch.from_numpy(obs).float(),
            action=action_tensor,
            reward=reward,
            done=is_learning_trigger
        )
        
        if is_learning_trigger:
            print(f"🔄 Segment Update (Step {step}) | Segment Reward: {current_segment_reward:.1f} | Total: {total_reward:.1f}")
            current_segment_reward = 0
            
            # 4. Visualization Snapshot (Skipping render, just logic tree if possible)
            # We can still save the tree to a file, user can download it
            try:
                device = agent.model.get_device()
                obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
                rtg_tensor = torch.tensor([[[rtg]]]).to(device)
                
                active_trace = get_decision_trace(agent.model, obs_tensor, rtg_tensor)
                graph = render_decision_tree(active_trace)
                if graph:
                     # Save without viewing
                    graph.render("agent_logic_tree", format="png", cleanup=True, view=False)
            except Exception as e:
                pass # Silent fail on visualization to keep training running
            
            # Explicit save every 10k steps for safety
            if step % 10000 == 0:
                print("💾 Safety checkpoint...")
                agent.save_weights()

            obs = next_obs
        
        if step % 1000 == 0:
            metrics_msg = f"Step {step} | Reward: {reward:.4f} | Curiosity: {curiosity.item():.4f}"
            print(metrics_msg)

except Exception as e:
    print(f"❌ Error during training: {e}")
finally:
    print("🛑 Loop finished or interrupted. Saving final weights...")
    agent.save_weights()
    print("✅ Final weights saved.")
    env.close()
