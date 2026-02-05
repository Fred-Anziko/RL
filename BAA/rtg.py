import torch
import torch.nn as nn
from .bayesian import BayesianLinear

class RTGAwareRouter(nn.Module):
    """ A router that conditions splits on both state embeddings and RTG values. 
    Inputs:
        - embed_dim: Dimension of state embeddings
    Outputs:
        - go_right_prob: Probability of routing to the right child node
    """
    def __init__(self, embed_dim):
        super().__init__()
        # Use BayesianLinear for consistent epistemic uncertainty
        self.router_gate = BayesianLinear(embed_dim + 1, 1)

    def forward(self, x, rtg):
        """ Forward pass for RTG-aware routing.
        Inputs:
            - x: State embeddings [batch, seq_len, embed_dim]
            - rtg: Reward-to-Go values [batch, seq_len, 1]
        Outputs:
            - go_right_prob: Probability (and KL divergence)
        """
        combined = torch.cat([x, rtg], dim=-1)
        logits = self.router_gate(combined)
        p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
        kl = self.router_gate.kl_divergence()
        return p, kl

class ConditionalSoftDecisionTree(nn.Module):
    """ A Differentiable Decision Tree that conditions splits on RTG values.
    Inputs:
        - embed_dim: Dimension of input embeddings
        - depth: Depth of the decision tree
    Outputs:
        - output: Weighted combination of leaf node values
        - node_trace: Dictionary containing routing probabilities for explainability
    """
    def __init__(self, embed_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_nodes = self.num_leaves - 1
        self.routers = nn.ModuleList([
            RTGAwareRouter(embed_dim) for _ in range(self.num_nodes)
        ])
        self.leaves = nn.Parameter(torch.randn(self.num_leaves, embed_dim))

    def forward(self, x, rtg):
        """Forward pass through the RTG-aware Soft Decision Tree.
        Inputs:
            - x: Input tensor of shape [batch, seq_len, embed_dim]
            - rtg: Reward-to-Go tensor of shape [batch, seq_len, 1]
        Outputs:
            - output: Output tensor of shape [batch, seq_len, embed_dim]
            - node_trace: Dictionary with routing probabilities for explainability
        """
        batch_size, seq_len, _ = x.shape
        
        node_probs = []
        node_trace = {}
        total_kl = torch.tensor(0.0, device=x.device)
        
        for i, router in enumerate(self.routers):
            p, kl = router(x, rtg)
            node_probs.append(p)
            total_kl += kl
            node_trace[f"node_{i}"] = {"go_right_prob": p, "go_left_prob": 1-p}
            
        node_probs_stack = torch.stack(node_probs)
        
        # FIX Issue 13: Build mu without in-place operations to preserve autograd
        mu_values = [torch.ones(batch_size, seq_len, 1).to(x.device)]
        
        for i in range(self.num_nodes):
            p_right = node_probs_stack[i].squeeze(-1)
            mu_left = mu_values[i] * (1 - p_right).unsqueeze(-1)
            mu_right = mu_values[i] * p_right.unsqueeze(-1)
            mu_values.append(mu_left)
            mu_values.append(mu_right)
            
        mu = torch.cat(mu_values, dim=-1)
        leaf_probs = mu[:, :, -self.num_leaves:].unsqueeze(-1)
        
        output = torch.matmul(leaf_probs.transpose(-2, -1), self.leaves.unsqueeze(0).unsqueeze(0))
        return output.squeeze(-2), node_trace, total_kl