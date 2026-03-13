import torch
import torch.nn as nn
import numpy as np
import random
from BAA.ffn import TreeTransformerBlock
from BAA.hindsight import HindsightRelabeler


# ---------------------------------------------------------------------------
# Prioritized Sequence Replay Buffer
# ---------------------------------------------------------------------------

class PrioritizedSequenceReplayBuffer:
    """
    Experience replay buffer with two improvements over the original:

    1. Multi-step sequences: stores overlapping windows of length `window_size`
       instead of individual timesteps. This lets the Transformer and RoPE attention
       learn temporal dependencies during training, not just at inference time.

    2. Prioritized sampling: new transitions are assigned maximum priority so they
       are sampled first. After each training step, priorities are updated with
       per-sample action losses (higher loss = more surprising = sample more often).
       Sampling probability: P(i) = priority_i^alpha / sum(priority_j^alpha).

    Inputs:
        - capacity: maximum number of windows to store
        - window_size: length of each stored sequence (default 8)
        - alpha: priority exponent — 0 = uniform, 1 = full prioritization
    """

    def __init__(self, capacity=100000, window_size=8, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.capacity = capacity
        self.window_size = window_size
        self.alpha = alpha
        self._max_priority = 1.0

    def __len__(self):
        return len(self.buffer)

    def push(self, states, actions, rtg):
        """
        Accepts an episode-length tensor [seq_len, dim] and slices it into
        overlapping windows of size `window_size`. Each window is stored as
        a separate entry [1, window_size, dim]. Short episodes are zero-padded
        on the right to reach window_size.

        Also accepts legacy 3D input [batch, 1, dim] for backward compatibility.
        """
        states = states.detach().cpu()
        actions = actions.detach().cpu()
        rtg = rtg.detach().cpu()

        # Normalize to 2D [seq_len, dim]
        if states.dim() == 3:
            states = states.squeeze(1)
            actions = actions.squeeze(1)
            rtg = rtg.squeeze(1)
        if rtg.dim() == 1:
            rtg = rtg.unsqueeze(-1)

        seq_len = states.shape[0]
        state_dim = states.shape[-1]
        action_dim = actions.shape[-1]
        rtg_dim = rtg.shape[-1]

        # Create overlapping windows. For episodes shorter than window_size,
        # one padded window is created.
        n_windows = max(1, seq_len - self.window_size + 1)
        for start in range(n_windows):
            end = start + self.window_size
            if end > seq_len:
                pad = end - seq_len
                s_win = torch.cat([states[start:], torch.zeros(pad, state_dim)], dim=0)
                a_win = torch.cat([actions[start:], torch.zeros(pad, action_dim)], dim=0)
                r_win = torch.cat([rtg[start:], torch.zeros(pad, rtg_dim)], dim=0)
            else:
                s_win = states[start:end]
                a_win = actions[start:end]
                r_win = rtg[start:end]

            entry = {
                "states":  s_win.unsqueeze(0),   # [1, window_size, state_dim]
                "actions": a_win.unsqueeze(0),
                "rtg":     r_win.unsqueeze(0),
            }

            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
                self.priorities.pop(0)

            self.buffer.append(entry)
            self.priorities.append(self._max_priority)

    def sample(self, batch_size):
        """
        Proportional priority sampling. Returns a list of sample dicts and
        the buffer indices used, so priorities can be updated after training.
        """
        n = len(self.buffer)
        actual_size = min(batch_size, n)

        probs = np.array(self.priorities, dtype=np.float64) ** self.alpha
        probs /= probs.sum()

        indices = np.random.choice(n, size=actual_size, replace=False, p=probs)
        samples = [self.buffer[i] for i in indices]

        # Filter to the most common shape in this batch to keep cat() valid
        if samples:
            ref_shape_s = samples[0]["states"].shape
            ref_shape_a = samples[0]["actions"].shape
            ref_shape_r = samples[0]["rtg"].shape
            valid = [
                (s, i) for s, i in zip(samples, indices)
                if s["states"].shape == ref_shape_s
                and s["actions"].shape == ref_shape_a
                and s["rtg"].shape == ref_shape_r
            ]
            if valid:
                samples, indices = zip(*valid)
                return list(samples), list(indices)

        return samples, list(indices)

    def update_priorities(self, indices, losses):
        """
        Update priorities for sampled entries using per-sample losses.
        Higher loss = more surprising transition = sample more frequently.

        indices: list of buffer indices from the last sample() call
        losses:  list or array of scalar loss values (one per sample)
        """
        for idx, loss in zip(indices, losses):
            if 0 <= idx < len(self.priorities):
                priority = float(abs(loss)) + 1e-6
                self.priorities[idx] = priority
                self._max_priority = max(self._max_priority, priority)


# ---------------------------------------------------------------------------
# Routing Tracker — for automated pruning
# ---------------------------------------------------------------------------

class RoutingTracker:
    """
    Tracks the time-averaged routing probability for every (layer, node) pair.

    When a node's mean P(Right) stays above `confidence_threshold` (or below
    1 - threshold) for at least `window_size` observations, it is considered
    "confident" and a candidate for automatic freezing. This progressively
    simplifies the tree as it learns, reducing inference cost and improving
    interpretability.
    """

    def __init__(self, confidence_threshold=0.95, window_size=100):
        self.threshold = confidence_threshold
        self.window_size = window_size
        # Lazy-initialized on first update
        self._sums = {}    # (layer_idx, node_idx) -> float
        self._counts = {}  # (layer_idx, node_idx) -> int
        self._frozen = set()  # already-pruned nodes

    def update(self, routing_trace):
        """Update running averages from a routing trace produced by AgenticDTT.forward()."""
        for layer_idx, layer_data in enumerate(routing_trace):
            for layer_name, tree_dec in layer_data.items():
                for node_name, probs in tree_dec.items():
                    node_idx = int(node_name.split("_")[1])
                    key = (layer_idx, node_idx)
                    p = probs["go_right_prob"]
                    if isinstance(p, torch.Tensor):
                        p = p.detach().mean().item()
                    self._sums[key] = self._sums.get(key, 0.0) + p
                    self._counts[key] = self._counts.get(key, 0) + 1

    def get_prune_candidates(self):
        """
        Return list of (layer_idx, node_id, direction) for nodes that have
        reached confident, consistent routing and have not yet been frozen.
        """
        candidates = []
        for key, count in self._counts.items():
            if count < self.window_size or key in self._frozen:
                continue
            mean_p = self._sums[key] / count
            layer_idx, node_idx = key
            if mean_p > self.threshold:
                candidates.append((layer_idx, node_idx, "right"))
            elif mean_p < (1.0 - self.threshold):
                candidates.append((layer_idx, node_idx, "left"))
        return candidates

    def mark_frozen(self, layer_idx, node_idx):
        """Record that this node has been frozen so it won't be re-queued."""
        self._frozen.add((layer_idx, node_idx))


# ---------------------------------------------------------------------------
# AgenticDTT — main model
# ---------------------------------------------------------------------------

class AgenticDTT(nn.Module):
    """
    Agentic Differentiable Decision Transformer with integrated Soft Decision Trees.

    Inputs:
        - state_dim: Dimension of input states
        - action_dim: Dimension of output actions
        - embed_dim: Embedding dimension for transformer
        - n_layers: Number of transformer layers
    Outputs:
        - action: Predicted action tensor
        - state_pred: Reconstructed state tensor
        - value: Predicted value tensor
        - routing_trace: List of routing probabilities for explainability
    """

    def __init__(self, state_dim, action_dim, embed_dim=256, n_layers=6):
        super().__init__()

        self.embed_state = nn.Linear(state_dim, embed_dim)
        self.embed_action = nn.Linear(action_dim, embed_dim)
        self.embed_reward = nn.Linear(1, embed_dim)

        self.blocks = nn.ModuleList([
            TreeTransformerBlock(embed_dim, num_heads=8, use_cross_attn=True)
            for _ in range(n_layers)
        ])

        self.policy_head = nn.Linear(embed_dim, action_dim)
        self.reconstruction_head = nn.Linear(embed_dim, state_dim)
        self.value_head = nn.Linear(embed_dim, 1)

        self.replay_buffer = PrioritizedSequenceReplayBuffer()
        self.routing_tracker = RoutingTracker()

    def get_device(self):
        return next(self.parameters()).device

    def forward(self, states, actions=None, rewards_to_go=None, external_context=None):
        """
        Forward pass through the AgenticDTT model.

        Inputs:
            - states: [batch, seq_len, state_dim]
            - actions: [batch, seq_len, action_dim]
            - rewards_to_go: [batch, seq_len, 1] — used as cross-attention context
            - external_context: [batch, any_seq, embed_dim] — optional goal/directive
        Outputs:
            - dict with action, state_pred, value, routing_trace, kl_div, attention_maps
        """
        device = self.get_device()
        states = states.to(device)

        x = self.embed_state(states)
        if actions is not None:
            x = x + self.embed_action(actions.to(device))

        context = None
        if rewards_to_go is not None:
            context = self.embed_reward(rewards_to_go.to(device))
        if external_context is not None:
            ext = external_context.to(device)
            context = ext if context is None else torch.cat([context, ext], dim=1)

        full_routing_trace = []
        full_attention_maps = []
        total_kl = torch.tensor(0.0, device=device)

        for i, block in enumerate(self.blocks):
            x, tree_trace, layer_kl, attn_weights = block(x, context=context)
            full_routing_trace.append({f"layer_{i}": tree_trace})
            full_attention_maps.append(attn_weights.detach().cpu())
            total_kl = total_kl + layer_kl

        return {
            "action":        torch.tanh(self.policy_head(x)),
            "state_pred":    self.reconstruction_head(x),
            "value":         self.value_head(x),
            "routing_trace": full_routing_trace,
            "kl_div":        total_kl,
            "attention_maps": full_attention_maps,
        }

    def get_logic_path(self, states, actions=None, rtg=None):
        """Retrieve routing probabilities (logic path) for given inputs."""
        device = self.get_device()
        states = states.to(device)
        if actions is not None:
            actions = actions.to(device)
        if rtg is not None:
            rtg = rtg.to(device)
        with torch.no_grad():
            output = self.forward(states, actions, rtg)
            trace = output["routing_trace"]
            for layer_data in trace:
                for layer_name, tree_dec in layer_data.items():
                    for node_name, probs in tree_dec.items():
                        tree_dec[node_name] = {
                            k: v.cpu().numpy() if isinstance(v, torch.Tensor) else v
                            for k, v in probs.items()
                        }
            return trace

    def on_episode_finish(self, raw_episode_data):
        """
        Process and store episode data into the replay buffer with Hindsight Relabeling.
        Both the original and hindsight samples are pushed, each as multi-step windows.

        Inputs:
            - raw_episode_data: List of (state, action, reward) tuples
        """
        if not raw_episode_data:
            return
        relabeler = HindsightRelabeler()
        samples = relabeler.relabel_episode(raw_episode_data)
        if not samples:
            return
        for sample in samples:
            self.replay_buffer.push(sample["states"], sample["actions"], sample["rtg"])

    def record_routing(self, routing_trace):
        """Feed a routing trace into the RoutingTracker for auto-prune statistics."""
        self.routing_tracker.update(routing_trace)

    def check_auto_prune(self):
        """
        Check if any decision nodes have become consistently confident and freeze them.
        Returns a list of (layer_idx, node_id, direction) tuples for nodes that were pruned.
        """
        candidates = self.routing_tracker.get_prune_candidates()
        pruned = []
        for layer_idx, node_idx, direction in candidates:
            success = self.freeze_node(layer_idx, node_idx, direction)
            if success:
                self.routing_tracker.mark_frozen(layer_idx, node_idx)
                pruned.append((layer_idx, node_idx, direction))
                print(f"  Auto-pruned: layer {layer_idx}, node {node_idx} → {direction}")
        return pruned

    def freeze_node(self, layer_idx, node_id, direction="left"):
        """
        Freeze a specific decision node by forcing its routing direction.
        Used for both surgical pruning (Neural Debugger) and automated pruning.
        """
        if layer_idx >= len(self.blocks):
            print(f"  Freeze failed: layer {layer_idx} out of bounds (max: {len(self.blocks) - 1})")
            return False

        tree = self.blocks[layer_idx].decision_tree_ffn
        if node_id >= len(tree.routers):
            print(f"  Freeze failed: node {node_id} out of bounds (max: {len(tree.routers) - 1})")
            return False

        router = tree.routers[node_id]
        with torch.no_grad():
            if direction == "right":
                router.bias_mu.fill_(50.0)
            else:
                router.bias_mu.fill_(-50.0)
            router.weight_mu.requires_grad = False
            router.weight_rho.requires_grad = False
            router.bias_mu.requires_grad = False
            router.bias_rho.requires_grad = False

        if router.weight_mu.requires_grad or router.bias_mu.requires_grad:
            print(f"  WARNING: Frozen node still has gradients enabled!")
            return False

        return True

    def step_annealing(self):
        """Decay temperature across all decision tree blocks."""
        return [block.step_annealing() for block in self.blocks]

    def set_sparsity(self, enabled, threshold=0.01, top_k=2):
        """Control sparsity for all layers."""
        for block in self.blocks:
            block.set_sparsity(enabled, threshold, top_k)
