#This file implements the KV-Cache logic and the Engineered Model configuration. It handles the transition from O(N^2) to O(N) processing.

import torch
import torch.nn as nn
import torch.nn.functional as F

class KVCacheTransformer(nn.Module):
    def __init__(self, vocab_size=50257, d_model=256, n_heads=8):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.d_model = d_model
        self.n_heads = n_heads
        
        # Simple projection layers to simulate attention keys and values
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

    def forward_naive(self, x):
        """Standard Forward Pass: O(N^2) complexity (Recomputes everything)"""
        x = self.embedding(x)
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # Standard attention mechanism
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model**0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        return torch.matmul(attn_probs, v)

    def forward_engineered(self, x, kv_cache=None):
        """
        Engineered Forward Pass: O(N) complexity using KV-Caching.
        Only computes the attention for the NEWEST token in the sequence.
        """
        x_emb = self.embedding(x)
        
        # If we have a cache, we only care about the last token's embedding
        if kv_cache is not None:
            current_x = x_emb[:, -1:, :] 
        else:
            current_x = x_emb

        q = self.q_proj(current_x)
        k = self.k_proj(current_x)
        v = self.v_proj(current_x)

        if kv_cache is not None:
            prev_k, prev_v = kv_cache
            # Concatenate new Key/Value with the historical cache
            k = torch.cat([prev_k, k], dim=1)
            v = torch.cat([prev_v, v], dim=1)

        # Attention calculation using the full history (k, v) but only current query (q)
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) / (self.d_model**0.5)
        attn_probs = F.softmax(attn_weights, dim=-1)
        output = torch.matmul(attn_probs, v)
        
        return self.out_proj(output), (k, v)
