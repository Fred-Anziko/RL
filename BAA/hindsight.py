import torch
import random


class HindsightRelabeler:
    """
    Implements true Hindsight Experience Replay (HER) with goal relabeling.

    For each completed episode, produces TWO training samples:
      1. The original trajectory with its intended RTG goal (what the agent aimed for).
      2. A hindsight-relabeled trajectory where the RTG at every step is recomputed
         as if the agent's actual final outcome was always the intended goal.
         This turns every episode — including failures — into a "successful" example
         of achieving a different target, solving the sparse-reward problem.

    Strategy: "final" — the hindsight goal is the actual cumulative return of the episode.

    inputs:
        - gamma: Discount factor for future rewards
    outputs:
        - list of training samples (dicts), one original + one hindsight per episode
    """

    def __init__(self, gamma=1.0):
        self.gamma = gamma

    def compute_rtg(self, rewards_tensor, override_total=None):
        """
        Calculates Reward-to-Go for every timestep.

        If override_total is provided, it is used as the RTG at t=0 and the
        per-step values are scaled proportionally (hindsight re-goal).

        rewards_tensor: [seq_len]
        override_total: float or None
        Returns: rtg_buffer of shape [seq_len, 1]
        """
        device = rewards_tensor.device
        seq_len = rewards_tensor.size(0)
        rtg_buffer = torch.zeros(seq_len, 1, device=device)

        running_return = 0.0
        for t in reversed(range(seq_len)):
            running_return = float(rewards_tensor[t]) + self.gamma * running_return
            rtg_buffer[t, 0] = running_return

        if override_total is not None and abs(running_return) > 1e-8:
            scale = override_total / running_return
            rtg_buffer = rtg_buffer * scale

        return rtg_buffer

    def relabel_episode(self, episode_buffer):
        """
        Takes a raw episode trace and returns a list of training samples.

        episode_buffer: List of (state, action, reward) tuples

        Returns: list of dicts, each with keys: "states", "actions", "rtg", "rewards"
          - Index 0: original trajectory (intended goal)
          - Index 1: hindsight-relabeled trajectory (actual outcome as goal)
        """
        if not episode_buffer:
            return None

        states, actions, rewards = zip(*episode_buffer)

        device = states[0].device
        states_tensor = torch.stack(states).to(device)
        actions_tensor = torch.stack(actions).to(device)

        reward_list = []
        for r in rewards:
            if isinstance(r, torch.Tensor):
                reward_list.append(r.item() if r.dim() == 0 else r.squeeze().item())
            else:
                reward_list.append(float(r))

        rewards_tensor = torch.tensor(reward_list, dtype=torch.float32, device=device)

        # --- Sample 1: Original trajectory ---
        # RTG computed from actual rewards — this is what the agent was trying to achieve.
        original_rtg = self.compute_rtg(rewards_tensor)

        original_sample = {
            "states": states_tensor,
            "actions": actions_tensor,
            "rtg": original_rtg,
            "rewards": rewards_tensor,
            "is_hindsight": False,
        }

        # --- Sample 2: Hindsight-relabeled trajectory ---
        # The actual total return becomes the new "goal" RTG at t=0.
        # Every intermediate RTG is rescaled as if the agent always intended
        # to reach this outcome. This is the core HER re-goal operation.
        actual_total_return = float(rewards_tensor.sum())
        hindsight_rtg = self.compute_rtg(rewards_tensor, override_total=actual_total_return)

        hindsight_sample = {
            "states": states_tensor,
            "actions": actions_tensor,
            "rtg": hindsight_rtg,
            "rewards": rewards_tensor,
            "is_hindsight": True,
        }

        return [original_sample, hindsight_sample]
