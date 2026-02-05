import torch

class HindsightRelabeler:
    """
    Implements Hindsight Experience Replay (HER) by recalculating the Reward-to-Go (RTG)
    based on actual episode outcomes.
    inputs:
        - gamma: Discount factor for future rewards
    outputs:
        - hindsight_sample: Dictionary containing relabeled episode data
    """
    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def compute_actual_rtg(self, rewards_tensor):
        """
        Calculates the true 'Reward-to-Go' for every timestep.
        Handle both single sequences and batched episodes
        rewards_tensor: [seq_len] or [seq_len, 1] or [batch, seq_len]
        Returns: rtg_buffer of same shape as rewards_tensor
        """
        # Capture device early before any operations
        device = rewards_tensor.device
        is_batched = rewards_tensor.dim() > 1 and rewards_tensor.size(0) > 1 and rewards_tensor.size(-1) != 1
        
        if is_batched:
            # Batched processing: [batch, seq_len]
            batch_size, seq_len = rewards_tensor.shape
            rtg_buffer = torch.zeros(batch_size, seq_len, 1, device=device)
            
            for b in range(batch_size):
                running_return = 0
                for t in reversed(range(seq_len)):
                    adjusted_reward = rewards_tensor[b, t]
                    running_return = adjusted_reward + self.gamma * running_return
                    rtg_buffer[b, t, 0] = running_return
        else:
            # Single sequence: [seq_len] or [seq_len, 1]
            if rewards_tensor.dim() > 1:
                rewards_tensor = rewards_tensor.squeeze(-1)
                
            seq_len = rewards_tensor.size(0)
            rtg_buffer = torch.zeros(seq_len, 1, device=device)
            running_return = 0
            
            # Backward pass to calculate future returns
            for t in reversed(range(seq_len)):
                adjusted_reward = rewards_tensor[t]
                running_return = adjusted_reward + self.gamma * running_return
                rtg_buffer[t, 0] = running_return
        return rtg_buffer

    def relabel_episode(self, episode_buffer):
        """
        Takes a raw episode trace and creates a Hindsight-Corrected training sample.
        episode_buffer: List of (state, action, reward)
        Returns a dictionary with keys: "states", "actions", "rtg", "rewards"
        """
        if not episode_buffer:
            return None
            
        states, actions, rewards = zip(*episode_buffer)
        
        # Convert to tensors immediately and ensure same device
        # We assume the first state's device is the target
        device = states[0].device
        
        states_tensor = torch.stack(states).to(device)
        actions_tensor = torch.stack(actions).to(device)
        
        # Create rewards tensor directly on target device
        reward_list = []
        for r in rewards:
            if isinstance(r, torch.Tensor):
                reward_list.append(r.item() if r.dim() == 0 else r.squeeze().item())
            else:
                reward_list.append(float(r))
        # Create tensor directly on target device to avoid device transfers
        rewards_tensor = torch.tensor(reward_list, dtype=torch.float32, device=device)
        
        # 1. Calculate the REALITY (What actually happened)
        actual_rtgs = self.compute_actual_rtg(rewards_tensor)
        
        # 2. Create the Hindsight Sample
        hindsight_sample = {
            "states": states_tensor,
            "actions": actions_tensor,
            "rtg": actual_rtgs,
            "rewards": rewards_tensor
        }
        
        return hindsight_sample