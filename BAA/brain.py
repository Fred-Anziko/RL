import torch
import numpy as np
import random
from .loss import DTTLossEngine
from .curiosity import CuriosityEngine

class AgenticBrain:
    """
    A 'full-stack' self-regulating brain that manages both 
    learning objectives and autonomous action selection.
    """
    def __init__(self, model, lr=1e-4, loss_engine=None, curiosity_engine=None, optimizer=None, training_step=0,action_dim=None):
        self.model = model
        self.loss_engine = loss_engine if loss_engine else DTTLossEngine()
        self.curiosity_engine = curiosity_engine if curiosity_engine else CuriosityEngine(model)
        self.optimizer = optimizer if optimizer else torch.optim.AdamW(model.parameters(), lr=lr)
        self.training_step = training_step  # Track to reduce overfitting to early random experiences
        self.action_dim = action_dim if action_dim is not None else model.policy_head.out_features  # Get action dim from model

    def act(self, state, rtg, explore=True, episode_num=1,start_epsilon=0.5,min_epsilon=0.05,decay_rate=0.9994):
        """
        Autonomous Action Selection: Determining exploration via Curiosity.
        state: Current observed state tensor
        rtg: Reward-to-Go tensor
        explore: Whether to apply exploration strategy
        episode_num: Current episode number for decay scheduling
        Returns: action tensor, decision trace, curiosity score
        """
        # Ensure state/rtg are on correct device
        device = self.model.get_device()
        state = state.to(device)
        rtg = rtg.to(device)

        with torch.no_grad():
            output = self.model(state, rewards_to_go=rtg)
            action = output['action']
            trace = output['routing_trace'] # Extract trace directly from output
            kl_div = output.get('kl_div')  # Extract Bayesian uncertainty
        
        curiosity_score = self.curiosity_engine.compute_intrinsic_reward(trace, kl_div=kl_div)
        
        if explore:
            # === OPTIMIZED FOR 2000 EPISODES ===
            start_epsilon = start_epsilon   # Initial exploration probability
            min_epsilon = min_epsilon    # Minimum "floor" for exploration
            decay_rate = decay_rate   # Reaches min_epsilon around Episode 1500
            
            # 1. Calculate the base decay
            base_epsilon = max(min_epsilon, start_epsilon * (decay_rate ** (episode_num - 1)))
            
            # 2. Add Curiosity-Driven Boost
            # Curiosity keeps exploration high in states where the tree is confused
            epsilon = base_epsilon + (curiosity_score.item() * 0.1)
            epsilon = min(epsilon, 1.0)
            
            if np.random.random() < epsilon:
                # Stochastic discovery: Random action (dynamic based on action dim)
                action = torch.tanh(torch.randn(1, 1, self.action_dim) * 2.0)  # Random scaled actions
        
        return action, trace, curiosity_score

    def train_on_buffer(self, batch_size=32):
        """
        Core Self-Regulation: Samples from the relabeled replay buffer 
        to update the model's neural logic.
        batch_size: Number of samples to draw from replay buffer
        Returns: training results dictionary
        """
        # FIX Issue 13: Check for sufficient data in replay buffer
        if len(self.model.replay_buffer) < batch_size:
            if np.random.random() < 0.05:  # Log occasionally
                print(f"⚠️  BAA training skipped: insufficient data in replay buffer ({len(self.model.replay_buffer)}/{batch_size})")
            return None
        # After that, train every step
        self.training_step += 1  
        batch = self.model.replay_buffer.sample(batch_size)
        device = self.model.get_device()
        
        # Experience Replay Unpacking
        # Clone to avoid autograd issues with shared storage
        states = torch.cat([x["states"].clone() for x in batch], dim=0).to(device).detach().requires_grad_(True)
        actions = torch.cat([x["actions"].clone() for x in batch], dim=0).to(device).detach().clone()
        rtgs = torch.cat([x["rtg"].clone() for x in batch], dim=0).to(device).detach().clone()
        reconstruction_target = states  # current states

        
        self.optimizer.zero_grad()
        
        # Forward pass through the decision backbone
        output = self.model(states, rewards_to_go=rtgs)
        
        # Loss calculation (Signal-Aware)
        results = self.loss_engine(
            model_output=output,
            target_actions=actions,
            target_states=reconstruction_target
        )
        
        results['loss'].backward()
        self.optimizer.step()
        
        results['active_mode'] = "Task-Mastery (Buffer Replay)"
        results['total_loss'] = results['loss'].item()
        return results

    def step_learning(self, experience_batch):
        """
        Live Signal Update: For immediate world-model adjustments.
        experience_batch: Dictionary with keys: "states", "actions", "rtg"
        Returns: training results dictionary
        """
        device = self.model.get_device()
        states = experience_batch.get("states").to(device)
        actions = experience_batch.get("actions")
        if actions is not None: actions = actions.to(device)
        rtg = experience_batch.get("rtg")
        if rtg is not None: rtg = rtg.to(device)
        
        self.optimizer.zero_grad()
        output = self.model(states, rewards_to_go=rtg)
        
        results = self.loss_engine(
            model_output=output,
            target_actions=actions,
            target_states=states
        )
        
        results['loss'].backward()
        self.optimizer.step()
        
        results['active_mode'] = "Signal-Direct (Live)"
        return results