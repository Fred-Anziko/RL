import gymnasium as gym
import torch
import numpy as np
from BAA.baa_interface import BAAAgent
from BAA.interpreter import get_decision_trace, render_decision_tree

# 1. Setup Environment
# 1. Setup Environment (Infinite Life Mode)
# terminate_when_unhealthy=False: prevents the env from stopping when the agent falls
# terminate_when_unhealthy=False: prevents the env from stopping when the agent falls
env = gym.make("Humanoid-v4", render_mode="human", terminate_when_unhealthy=False)

# 2. Initialize the Unified Agent
agent = BAAAgent(
    state_dim=376, 
    action_dim=17, 
    weight_path="humanoid_model.pt"
)

total_episodes_reward = []

# Initialize once
obs, info = env.reset()
rtg = 15000.0  # Set a realistic high-performance target
total_reward = 0
current_segment_reward = 0

print("🚀 Starting Infinite Life Training: Agent will NOT reset on falls.")

for step in range(5000000): # Increased horizon
    # 1. Get Action
    action_tensor, trace, curiosity = agent.get_action(obs, rtg)
    action = action_tensor.detach().cpu().numpy().flatten()
    
    # 2. Step the environment
    next_obs, reward, terminated, truncated, info = env.step(action)
    
    current_segment_reward += reward
    total_reward += reward
    
    # 3. Continuous Learning Logic
    # We never reset the environment ("Infinite Life"), but we need to tell 
    # the brain to "digest" the recent experience periodically.
    # We treat every 1000 steps as a "learning segment".
    # FIX: Ignore 'truncated' from env because we want to go beyond the TimeLimit
    is_learning_trigger = (step % 1000 == 0 and step > 0)
    
    agent.record_experience(
        session_id="infinite_life_01",
        state=torch.from_numpy(obs).float(),
        action=action_tensor,
        reward=reward,
        done=is_learning_trigger # Triggers training in BAA logic, but we continue physically
    )
    
    if is_learning_trigger:
        print(f"🔄 Segment Update (Step {step}) | Segment Reward: {current_segment_reward:.1f} | Total: {total_reward:.1f}")
        current_segment_reward = 0
        
        # 4. Visualization Snapshot
        # Generate a trace using the last observation and target RTG
        # We need to ensure obs is a tensor on the correct device
        device = agent.model.get_device()
        obs_tensor = torch.from_numpy(obs).float().unsqueeze(0).unsqueeze(0).to(device)
        rtg_tensor = torch.tensor([[[rtg]]]).to(device)
        
        active_trace = get_decision_trace(agent.model, obs_tensor, rtg_tensor)
        graph = render_decision_tree(active_trace)
        if graph:
            try:
                graph.render("agent_logic_tree", format="png", cleanup=True)
                print(f"📊 Logic tree visualization saved to agent_logic_tree.png")
            except Exception as e:
                if "ExecutableNotFound" in str(type(e)):
                    print(f"⚠️  Logic tree visualization failed: Graphviz 'dot' executable not found. Simulation continuing.")
                else:
                    print(f"⚠️  Logic tree visualization failed: {e}")
        obs = next_obs
    
    if step % 100 == 0:
        # Extract average KL from trace for visibility
        metrics_msg = f"Step {step} | Reward: {reward:.4f} | Curiosity: {curiosity.item():.4f}"
        print(metrics_msg)