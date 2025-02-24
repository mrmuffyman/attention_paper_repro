import torch
import torch.nn as nn
import torch.nn.functional as F

import math

VOCAB_SIZE = 32000

class Embedder(nn.Module):
    def __init__(self, model_dim=512, max_seq_length=2048):
        super().__init__()
        self.token_embedding = nn.Embedding(VOCAB_SIZE, model_dim)
        self.model_dim = model_dim
        
        # Create positional encoding matrix
        position = torch.arange(max_seq_length).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, model_dim, 2) * (-math.log(10000.0) / model_dim))
        pos_encoding = torch.zeros(max_seq_length, model_dim)
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        # Register as buffer so it moves to the correct device with the model
        self.register_buffer('pos_encoding', pos_encoding)

    def forward(self, x):
        # Get sequence length from input
        seq_len = x.size(1)
        # Token embeddings
        token_embeddings = self.token_embedding(x)
        # Add positional encodings
        position_encodings = self.pos_encoding[:seq_len, :]
        # Scale embeddings by sqrt(d_model) as in the paper
        return (token_embeddings * math.sqrt(self.model_dim)) + position_encodings

class AttentionLayer(nn.Module):
    def __init__(self, model_dim=512, key_dim=64):
        """
        Dimensions:
        - d_model: 512 (input dimension)
        - d_k: 64 (key/query dimension)
        - d_v: 64 (value dimension)
        """
        super().__init__()
        # Query: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        self.query_stack = nn.Sequential(
                    nn.Linear(model_dim,key_dim),
                    nn.ReLU()
                )
        # Key: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_k)
        self.key_stack = nn.Sequential(
                    nn.Linear(model_dim,key_dim),
                    nn.ReLU()
                )
        # Value: (batch_size, seq_len, d_model) -> (batch_size, seq_len, d_v)
        self.value_stack = nn.Sequential(
                    nn.Linear(model_dim,key_dim),
                    nn.ReLU()
                )
        self.key_dim = key_dim
        self.model_dim = model_dim
        
    def forward(self, x1, x2, apply_mask=False):
        """
        Args:
            x1: Query input (batch_size, seq_len, d_model)
            x2: Key/Value input (batch_size, seq_len, d_model)
            apply_mask: Whether to apply causal masking
            
        Returns:
            output: (batch_size, seq_len, d_v)
        """
        query = self.query_stack(x1)  # (batch_size, seq_len, d_k)
        key = self.key_stack(x2)      # (batch_size, seq_len, d_k)
        value = self.value_stack(x2)  # (batch_size, seq_len, d_v)

        # (batch_size, seq_len, seq_len)
        query_key_prod = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.key_dim)
        
        if apply_mask:
            seq_len = query_key_prod.shape[-1]
            mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool().to(query_key_prod.device)
            query_key_prod.masked_fill_(mask, float('-inf'))

        # (batch_size, seq_len, seq_len)
        softmax_qk = F.softmax(query_key_prod, dim=-1)
        # (batch_size, seq_len, d_v)
        output = torch.matmul(softmax_qk, value)
        return output

class MultiHeadAttention(nn.Module):
    def __init__(self, heads=8, model_dim=512):
        # Input X: dSeqLen x dModel
        super().__init__()
        assert model_dim % heads == 0, "model_dim must be divisible by heads, got %d and %d" % (model_dim, heads)
        self.attention_heads = nn.ModuleList([AttentionLayer(model_dim, model_dim // heads) for _ in range(heads)])
    def forward(self, x1, x2, apply_mask=False):
        # heads x dSeqLen x dV
        head_outputs = [head(x1, x2, apply_mask) for head in self.attention_heads]
        # dSeqLen x dModel
        return torch.concat(head_outputs, dim=-1)
    
class LayerNorm(nn.Module):
    def __init__(self, model_dim=512):
        """
        Layer normalization with learnable parameters gamma and beta.
        Normalizes across the last dimension (d_model = 512).
        """
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(model_dim))   # (d_model,)
        self.beta = nn.Parameter(torch.zeros(model_dim))   # (d_model,)
        self.eps = 1e-5

    def forward(self, x):
        """
        Args:
            x: Input tensor (batch_size, seq_len, d_model)
        Returns:
            Normalized tensor of same shape
        """
        mean = x.mean(dim=-1, keepdim=True)
        std = x.std(dim=-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

    
class Encoder(nn.Module):
    def __init__(self):
        # Input X: dSeqLen x dModel
        super().__init__()
        self.mha = MultiHeadAttention()
        self.lnorm_mha = LayerNorm()
        self.lnorm_ff = LayerNorm()

        self.ff = nn.Sequential(
                    nn.Linear(512,2048),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(2048, 512)
                )
    def forward(self, x):
        mha_output = self.lnorm_mha(self.mha(x, x) + x)
        ff_output = self.lnorm_ff(self.ff(mha_output) + mha_output)
        return ff_output

class Decoder(nn.Module):
    def __init__(self, heads=8, model_dim=512):
        # Input X: dSeqLen x dModel
        super().__init__()
        assert model_dim % heads == 0, "model_dim must be divisible by heads, got %d and %d" % (model_dim, heads)
        # Multihead self attention
        self.mhsa = MultiHeadAttention(heads, model_dim)
        # Multihead cross attention
        self.mhca = MultiHeadAttention(heads, model_dim)
        # Self attention layer norm
        self.lnorm_mhsa = LayerNorm(model_dim)
        # Cross attention layer norm
        self.lnorm_mhca = LayerNorm(model_dim)

        self.lnorm_ff = LayerNorm(model_dim)

        self.ff = nn.Sequential(
                    nn.Linear(model_dim,2048),
                    nn.ReLU(),
                    nn.Dropout(0.1),
                    nn.Linear(2048, model_dim)
                )
    # x is shape seqLen, dModel 
    # encoder_x is shape seqLen, dModel
    # assume x is already masked
    def forward(self, decoder_x, encoder_x, skip_cross_attn=False):
        multihead_self_attn_output = self.lnorm_mhsa(self.mhsa(decoder_x, decoder_x, apply_mask=True) + decoder_x)
        multihead_cross_attn_output = multihead_self_attn_output
        if (not skip_cross_attn):
            multihead_cross_attn_output = self.lnorm_mhca(self.mhca(multihead_self_attn_output, encoder_x) + multihead_self_attn_output)

        ff_output = self.lnorm_ff(self.ff(multihead_cross_attn_output) + multihead_cross_attn_output)
        return ff_output
    
