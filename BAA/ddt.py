import torch
import torch.nn as nn
from BAA.bayesian import BayesianLinear
import torch.nn.functional as F

class SoftDecisionTree(nn.Module):
    """
    A Differentiable Decision Tree that acts as the 'reasoning' 
    unit within the Transformer.
    Inputs:
        - input_dim: Dimension of input features
        - output_dim: Dimension of output features
        - depth: Depth of the tree (number of levels)
    Outputs:
        - output: Weighted combination of leaf node values
        - node_trace: Dictionary containing routing probabilities for explainability
    """
    def __init__(self, input_dim, output_dim, depth=3):
        super().__init__()
        self.depth = depth
        self.num_leaves = 2 ** depth
        self.num_nodes = self.num_leaves - 1
        self.temperature = 1.0  
        self.min_temp = 0.1
        self.annealing_rate = 0.999
        
        self.use_sparsity = False
        self.top_k = 2
        self.sparsity_threshold = 0.01
        
        # Internal routing nodes (linear splits)
        self.routers = nn.ModuleList([
            BayesianLinear(input_dim, 1) for _ in range(self.num_nodes)
        ])
        
        # Functional Leaves (Expert layers) - replaces constant parameters
        self.leaves = nn.ModuleList([
            BayesianLinear(input_dim, output_dim) for _ in range(self.num_leaves)
        ])

    def forward(self, x):
        """Forward pass through the Soft Decision Tree.
        Inputs:
            - x: Input tensor of shape [batch, seq_len, input_dim]
        Outputs:
            - output: Output tensor of shape [batch, seq_len, output_dim]
            - node_trace: Dictionary with routing probabilities for explainability
        """
        # x shape: [batch, seq_len, input_dim]
        batch_size, seq_len, _ = x.shape
        
        # Internal node probabilities (prob of going 'right')
        # node_probs shape: [num_nodes, batch, seq_len, 1]
        node_probs = []
        node_trace = {}
        kl_total = torch.tensor(0.0, device=x.device)
        for i, router in enumerate(self.routers):
            logits = router(x) / self.temperature
            # Accumulate KL divergence from each router
            kl_total += router.kl_divergence()
            
            p = torch.sigmoid(logits).clamp(1e-7, 1-1e-7)
            node_probs.append(p)
            node_trace[f"node_{i}"] = {"go_right_prob": p, "go_left_prob": 1-p}
        
        node_probs_stack = torch.stack(node_probs)
        
        # mu[i] is the probability of reaching node i
        # CRITICAL FIX: Build mu without in-place operations to preserve autograd
        mu_values = [torch.ones(batch_size, seq_len, 1).to(x.device)]  # mu[0]
        
        for i in range(self.num_nodes):
            p_right = node_probs_stack[i].squeeze(-1)  # [batch, seq_len]
            # mu[2*i + 1] = mu[i] * (1 - p_right)
            # mu[2*i + 2] = mu[i] * p_right
            mu_left = mu_values[i] * (1 - p_right).unsqueeze(-1)
            mu_right = mu_values[i] * p_right.unsqueeze(-1)
            mu_values.append(mu_left)
            mu_values.append(mu_right)
        
        # Concatenate to form complete mu tensor
        mu = torch.cat(mu_values, dim=-1)  # [batch, seq_len, 2^(depth+1) - 1]
        
        # Leaf probabilities are the last 2^depth entries in mu
        leaf_probs = mu[:, :, -self.num_leaves:] # [batch, seq_len, num_leaves]
        
        # 4. Top-K Sparsity (Optional)
        if self.use_sparsity:
            # Determine Top-K (default to 1 if not specified, usually set by user)
            k = getattr(self, "top_k", 2) 
            k = min(k, self.num_leaves)
            
            top_k_probs, top_k_indices = torch.topk(leaf_probs, k=k, dim=-1)
            
            # Mask out non-top-k probabilities
            mask = torch.zeros_like(leaf_probs).scatter_(-1, top_k_indices, 1.0)
            leaf_probs = leaf_probs * mask
            
            # Renormalize to ensure partition of unity
            leaf_probs = leaf_probs / (leaf_probs.sum(dim=-1, keepdim=True) + 1e-7)
        
        # 5. Vectorized Expert Pass (Functional Leaves)
        # We replace the Python 'for' loop with a single stacked Bayesian pass
        # to maximize GPU utilization and throughput.
        
        # Collect parameters from experts
        # Shape: [num_leaves, output_dim, input_dim]
        w_mu = torch.stack([l.weight_mu for l in self.leaves])
        w_rho = torch.stack([l.weight_rho for l in self.leaves])
        # Shape: [num_leaves, output_dim]
        b_mu = torch.stack([l.bias_mu for l in self.leaves])
        b_rho = torch.stack([l.bias_rho for l in self.leaves])
        
        # Vectorized Bayesian Sampling
        w_sigma = F.softplus(w_rho)
        b_sigma = F.softplus(b_rho)
        
        if self.training:
            w_eps = torch.randn_like(w_mu)
            b_eps = torch.randn_like(b_mu)
            weights = w_mu + w_sigma * w_eps
            biases = b_mu + b_sigma * b_eps
            
            # Vectorized KL Divergence update
            # Prior is assumed N(0, prior_sigma) same for all leaves
            prior_sigma = self.leaves[0].prior_sigma
            kl_w = torch.log(prior_sigma / w_sigma) + (w_sigma**2 + w_mu**2) / (2 * prior_sigma**2) - 0.5
            kl_b = torch.log(prior_sigma / b_sigma) + (b_sigma**2 + b_mu**2) / (2 * prior_sigma**2) - 0.5
            kl_total += kl_w.sum() + kl_b.sum()
        else:
            weights = w_mu
            biases = b_mu
            
        # Parallel Logic: Apply all experts to all tokens in one matmul
        # x: [batch, seq, in]
        # weights: [leaves, out, in]
        # biases: [leaves, out]
        # Output: [batch, seq, leaves, out]
        expert_outputs = torch.einsum('bsi, loi -> bslo', x, weights) + biases.view(1, 1, self.num_leaves, -1)
        
        # 6. Weighted Logic Fusion
        # leaf_probs: [batch, seq, leaves]
        # expert_outputs: [batch, seq, leaves, out]
        # Final output: [batch, seq, out]
        output = torch.einsum('bsl, bslo -> bso', leaf_probs, expert_outputs)
        
        return output, node_trace, kl_total
    
    def step_annealing(self):
        """ Decay temperature for sharper decisions. """
        self.temperature = max(self.min_temp, self.temperature * self.annealing_rate)
        return self.temperature

    def get_routing_trace(self, x):
        """Explainability wrapper - gradients not needed.
        Inputs:
            - x: Input tensor of shape [batch, seq_len, input_dim]
        Outputs:
            - trace: Dictionary with routing probabilities converted to CPU/Numpy
        """
        with torch.no_grad():
            _, trace, _ = self.forward(x)
            # Convert to CPU/Numpy for metadata
            for k, v in trace.items():
                trace[k] = {ik: iv.cpu().numpy() for ik, iv in v.items()}
        return trace