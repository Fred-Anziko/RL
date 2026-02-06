import torch.nn as nn
from BAA.ddt import SoftDecisionTree
from BAA.rope import RotaryMultiheadAttention

class TreeTransformerBlock(nn.Module):
    """
    A Hybrid Transformer Block where the standard MLP is replaced by a 
    Soft Decision Tree, and self-attention is optionally followed by cross-attention.
    Inputs:
        - embed_dim: Dimension of input embeddings
        - num_heads: Number of attention heads
        - tree_depth: Depth of the decision tree
        - use_cross_attn: Whether to include cross-attention layer
    Outputs:
        - output: Transformed embeddings
        - tree_trace: Routing probabilities from the decision tree for explainability
    """
    def __init__(self, embed_dim, num_heads, tree_depth=4, use_cross_attn=True):
        super().__init__()
        # 1. Self-Attention: Understanding internal sequence dynamics
        # Replaced standard attention with Euler-based Rotary Embeddings (RoPE)
        self.attention = RotaryMultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(embed_dim)
        
        # 2. Cross-Attention: Fusing external goals/directives (optional)
        self.use_cross_attn = use_cross_attn
        if use_cross_attn:
            self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
            self.norm_cross = nn.LayerNorm(embed_dim)
        
        # 3. The Decision Tree replaces the standard MLP/FFN for 'Reasoning'
        self.decision_tree_ffn = SoftDecisionTree(embed_dim, embed_dim, depth=tree_depth)
        self.norm2 = nn.LayerNorm(embed_dim)

    def forward(self, x, context=None):
        """
        x: Current sequence embedding [batch, seq_len, embed_dim]
        context: Optional external context (Goal, RTG latent, etc.)
        Outputs:
            - output: Transformed embeddings [batch, seq_len, embed_dim]
            - tree_trace: Routing probabilities from the decision tree for explainability
        """
        # 1. Self-Attention
        attn_output, attn_weights = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        
        # 2. Cross-Attention (Active if context provided)
        if self.use_cross_attn:
            if context is not None:
                # States (Q) attend to Context (K, V)
                cross_output, _ = self.cross_attn(x, context, context)
                x = self.norm_cross(x + cross_output)
            else:
                # Silent degradation - log occasionally (2% for consistency)
                import random
                if random.random() < 0.02:
                    print("⚠️  Cross-attention enabled but context is None")
        
        # 3. Decision Tree Segment
        tree_output, tree_trace, kl_div = self.decision_tree_ffn(x)
        x = self.norm2(x + tree_output)
        
        return x, tree_trace, kl_div, attn_weights

    def step_annealing(self):
        """ Delegate annealing to the inner decision tree. """
        return self.decision_tree_ffn.step_annealing()

    def set_sparsity(self, enabled, threshold=0.01, top_k=2):
        """ Configure sparsity for the decision tree. """
        self.decision_tree_ffn.use_sparsity = enabled
        self.decision_tree_ffn.sparsity_threshold = threshold
        self.decision_tree_ffn.top_k = top_k