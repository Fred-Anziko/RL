import torch
import torch.nn as nn
import numpy as np
import random
from BAA.ffn import TreeTransformerBlock
from BAA.hindsight import HindsightRelabeler

class ReplayBuffer:
    """ Experience Replay Buffer for storing and sampling agent experiences.
    Inputs:
        - capacity: Maximum number of transitions to store
    Outputs:
        - buffer: List of stored transitions
    """
    def __init__(self, capacity=100000):
        self.buffer = []
        self.capacity = capacity
    def __len__(self):
        return len(self.buffer)

    def push(self, states, actions, rtg):
        """
        Store transitions in the buffer. Use iterating to store individual items 
        so that sampling returns specific count of transitions, not batches.
        Inputs:
            - states: Tensor of shape [batch, seq_len, state_dim]
            - actions: Tensor of shape [batch, seq_len, action_dim]
            - rtg: Tensor of shape [batch, seq_len, 1]
        Outputs:
            - None
        """
        # Ensure cpu storage to save VRAM
        states = states.detach().cpu()
        actions = actions.detach().cpu()
        rtg = rtg.detach().cpu()
        
        # Ensure 3D shape [Batch, Seq, Dim]
        if states.dim() == 2:
            states = states.unsqueeze(1)
        if actions.dim() == 2:
            actions = actions.unsqueeze(1)
        if rtg.dim() == 2:
            rtg = rtg.unsqueeze(1)
        
        # Validation for consistency
        if states.dim() != 3 or states.shape[1] != 1:
            print(f"⚠️  ReplayBuffer specific warning: Pushed states shape {states.shape} != (B, 1, Dim)")
        if actions.dim() != 3 or actions.shape[1] != 1:
            print(f"⚠️  ReplayBuffer specific warning: Pushed actions shape {actions.shape} != (B, 1, Dim)")
            
        batch_size = states.shape[0]
        
        for i in range(batch_size):
            if len(self.buffer) >= self.capacity:
                self.buffer.pop(0)
            
            # Store with leading dimension 1 (e.g. [1, seq_len, dim])
            # so torch.cat works identically to before
            self.buffer.append({
                "states": states[i:i+1].clone(),
                "actions": actions[i:i+1].clone(),
                "rtg": rtg[i:i+1].clone()
            })

    def sample(self, batch_size):
        """
        Sample experiences from buffer with proper batching.
        Inputs:
            - batch_size: Number of transitions to sample
        Outputs:
            - samples: List of sampled transitions
        """
        samples = random.sample(self.buffer, min(len(self.buffer), batch_size))
        
        if not samples:
            return []
            
        # FIX Issue 5: Validate tensor shapes for consistency
        first_sample = samples[0]
        expected_state_shape = first_sample["states"].shape
        expected_action_shape = first_sample["actions"].shape
        expected_rtg_shape = first_sample["rtg"].shape
        
        for idx, sample in enumerate(samples):
            if sample["states"].shape != expected_state_shape:
                print(f"⚠️  Sample {idx} state shape {sample['states'].shape} doesn't match expected {expected_state_shape}")
                # Skip mismatched samples instead of crashing
                continue
            if sample["actions"].shape != expected_action_shape:
                print(f"⚠️  Sample {idx} action shape {sample['actions'].shape} doesn't match expected {expected_action_shape}")
                continue
            if sample["rtg"].shape != expected_rtg_shape:
                print(f"⚠️  Sample {idx} rtg shape {sample['rtg'].shape} doesn't match expected {expected_rtg_shape}")
                continue
        
        # FIX Issue 5: Filter out incompatible samples
        compatible_samples = [
            s for s in samples 
            if s["states"].shape == expected_state_shape 
            and s["actions"].shape == expected_action_shape
            and s["rtg"].shape == expected_rtg_shape
        ]
        
        if len(compatible_samples) < len(samples):
            print(f"⚠️  Filtered {len(samples) - len(compatible_samples)} incompatible samples from batch")
        
        return compatible_samples if compatible_samples else []

class AgenticDTT(nn.Module):
    """ Agentic Differentiable Decision Transformer with integrated
    Soft Decision Trees for reasoning.
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
        
        # Initialize blocks with Cross-Attention enabled for goal-conditioning
        self.blocks = nn.ModuleList([
            TreeTransformerBlock(embed_dim, num_heads=8, use_cross_attn=True) 
            for _ in range(n_layers)
        ])
        
        self.policy_head = nn.Linear(embed_dim, action_dim)
        self.reconstruction_head = nn.Linear(embed_dim, state_dim)
        self.value_head = nn.Linear(embed_dim, 1)
        
        self.replay_buffer = ReplayBuffer()

    def get_device(self):
        """ Utility to get the device of the model parameters. """
        return next(self.parameters()).device

    def forward(self, states, actions=None, rewards_to_go=None, external_context=None):
        """
        Forward pass through the AgenticDTT model.
        Inputs:        
            - states: [batch, seq_len, state_dim]
            - actions: [batch, seq_len, action_dim]
            - rewards_to_go: [batch, seq_len, 1] - Used as Cross-Attention context
            - external_context: [batch, any_seq, embed_dim] - Optional goal/directive
        Outputs:
            - action: Predicted action tensor
            - state_pred: Reconstructed state tensor
            - value: Predicted value tensor
            - routing_trace: List of routing probabilities for explainability
        """
        device = self.get_device()
        states = states.to(device)
        
        x = self.embed_state(states)
        if actions is not None:
            x = x + self.embed_action(actions.to(device))
            
        # Context Fusion: RTG or External Directives
        context = None
        if rewards_to_go is not None:
            # Attend to Reward-to-Go latents instead of simple addition
            context = self.embed_reward(rewards_to_go.to(device))
        
        if external_context is not None:
            # Overwrite or append with external context (directives)
            context = external_context.to(device) if context is None else torch.cat([context, external_context.to(device)], dim=1)
            
        full_routing_trace = []
        full_attention_maps = []
        total_kl = torch.tensor(0.0, device=device)
        for i, block in enumerate(self.blocks):
            # The heart of the change: States attend to Context
            x, tree_trace, layer_kl, attn_weights = block(x, context=context)
            full_routing_trace.append({f"layer_{i}": tree_trace})
            full_attention_maps.append(attn_weights.detach().cpu())
            total_kl += layer_kl
            
        return {
            "action": torch.tanh(self.policy_head(x)),
            "state_pred": self.reconstruction_head(x),
            "value": self.value_head(x),
            "routing_trace": full_routing_trace,
            "kl_div": total_kl,
            "attention_maps": full_attention_maps
        }

    def get_logic_path(self, states, actions=None, rtg=None):
        """
        Retrieve the routing probabilities (logic path) for given inputs.
        Inputs:
            - states: [batch, seq_len, state_dim]
            - actions: [batch, seq_len, action_dim]
            - rtg: [batch, seq_len, 1]
        Outputs:
            - trace: List of routing probabilities for each tree layer
        """
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

        The relabeler returns a list of samples: [original, hindsight].
        Both are pushed to the replay buffer so the model learns from the
        original intended goal AND the re-labeled achieved goal.

        Inputs:
            - raw_episode_data: List of (state, action, reward) tuples
        Outputs:
            - None
        """
        if not raw_episode_data:
            return
        relabeler = HindsightRelabeler()
        samples = relabeler.relabel_episode(raw_episode_data)
        if not samples:
            return
        for sample in samples:
            self.replay_buffer.push(sample["states"], sample["actions"], sample["rtg"])

    def freeze_node(self, layer_idx, node_id, direction="left"):
        """
        Freeze a specific decision node by forcing its routing direction.
        Used for surgical pruning in the Neural Debugger.
        """
        if layer_idx >= len(self.blocks): 
            print(f"⚠️  Freeze failed: layer {layer_idx} out of bounds (max: {len(self.blocks) - 1})")
            return False
            
        tree = self.blocks[layer_idx].decision_tree_ffn
        if node_id >= len(tree.routers): 
            print(f"⚠️  Freeze failed: node {node_id} out of bounds (max: {len(tree.routers) - 1})")
            return False
            
        router = tree.routers[node_id]
        with torch.no_grad():
            if direction == "right": 
                router.bias_mu.fill_(50.0)
                print(f"✓ Froze node {node_id} @ layer {layer_idx} -> RIGHT (bias_mu=50.0)")
            else: 
                router.bias_mu.fill_(-50.0)
                print(f"✓ Froze node {node_id} @ layer {layer_idx} -> LEFT (bias_mu=-50.0)")
            
            # Disable gradients for all Bayesian parameters
            router.weight_mu.requires_grad = False
            router.weight_rho.requires_grad = False
            router.bias_mu.requires_grad = False
            router.bias_rho.requires_grad = False
        
        # Verify gradients are actually disabled
        if router.weight_mu.requires_grad or router.bias_mu.requires_grad:
            print(f"⚠️  WARNING: Frozen node still has gradients enabled!")
            return False
            
        return True

    def step_annealing(self):
        """ Decay temperature across all decision tree blocks. """
        temps = []
        for block in self.blocks:
            temps.append(block.step_annealing())
        return temps

    def set_sparsity(self, enabled, threshold=0.01, top_k=2):
        """ Control sparsity for all layers. """
        for block in self.blocks:
            block.set_sparsity(enabled, threshold, top_k)