import torch
import torch.nn as nn
import math

class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=2048, base=10000, device=None):
        super().__init__()
        
        self.dim = dim
        self.base = base
        # Precompute frequencies
        # inv_freq shape: [dim/2]
        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32).float().to(device) / dim))
        self.register_buffer("inv_freq", inv_freq)

        # Build cache for large sequences
        self.max_seq_len_cached = max_position_embeddings
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        
        # Consistent with llama/opt style rotations:
        # We need cos and sin for [seq_len, head_dim]
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0), persistent=False)

    def forward(self, x, seq_len=None):
        # x: [batch, num_heads, seq_len, head_dim]
        if seq_len > self.max_seq_len_cached:
            self._update_cos_sin_cache(seq_len, device=x.device, dtype=x.dtype)
            
        return (
            self.cos_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
            self.sin_cached[:, :, :seq_len, ...].to(dtype=x.dtype),
        )

    def _update_cos_sin_cache(self, seq_len, device, dtype):
        # Reset cache for larger sequence
        self.max_seq_len_cached = seq_len
        t = torch.arange(self.max_seq_len_cached, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        self.register_buffer("cos_cached", emb.cos().unsqueeze(0).unsqueeze(0).to(device), persistent=False)
        self.register_buffer("sin_cached", emb.sin().unsqueeze(0).unsqueeze(0).to(device), persistent=False)

def rotate_half(x):
    # Split the last dimension into two halves
    x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [batch, num_heads, seq_len, head_dim]
    # cos, sin: [1, 1, seq_len, head_dim]
    # Apply rotation: (x * cos) + (rotate_half(x) * sin)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed

class RotaryMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, batch_first=True, dropout=0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.batch_first = batch_first
        
        assert self.head_dim * num_heads == self.embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        self.dropout = nn.Dropout(dropout)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        
        # Initialize RoPE
        self.rotary_emb = RotaryEmbedding(self.head_dim)

    def forward(self, query, key, value, attn_mask=None):
        # query, key, value shape: [batch, seq_len, embed_dim] if batch_first
        # If not batch_first, we transpose for calculation then transpose back?
        # Let's support batch_first primarily as used in BAA.
        
        if not self.batch_first:
            # Transpose to [batch, seq, dim] for easier handling
            query = query.transpose(0, 1)
            key = key.transpose(0, 1)
            value = value.transpose(0, 1)
            
        batch_size, seq_len, _ = query.size()
        
        # Project Q, K, V
        # Shape: [batch, seq_len, embed_dim] -> [batch, seq_len, num_heads, head_dim]
        # Then transpose to [batch, num_heads, seq_len, head_dim] for attention
        q = self.q_proj(query).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply RoPE to Q and K
        cos, sin = self.rotary_emb(v, seq_len=seq_len)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        
        # Scaled Dot-Product Attention
        # (batch, heads, seq, head_dim) @ (batch, heads, head_dim, seq) -> (batch, heads, seq, seq)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        if attn_mask is not None:
             attn_weights = attn_weights.masked_fill(attn_mask == 0, float('-inf'))
             
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # (batch, heads, seq, seq) @ (batch, heads, seq, head_dim) -> (batch, heads, seq, head_dim)
        attn_output = torch.matmul(attn_weights, v)
        
        # Reassemble heads
        # Tranpose and reshape back to [batch, seq, embed_dim]
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.embed_dim)
        
        # Final projection
        output = self.out_proj(attn_output)
        
        if not self.batch_first:
            output = output.transpose(0, 1)
            
        return output, attn_weights
