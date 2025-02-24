from nn_architecture import *

import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderDecoder(nn.Module):
    def __init__(self, heads=6):
        super().__init__()
        self.embedder = Embedder()
        self.encoder_stack = nn.Sequential(*[Encoder() for _ in range(heads)])
        self.decoder_stack = nn.ModuleList([Decoder() for _ in range(heads)])
        self.linear_projection = nn.Linear(512, VOCAB_SIZE)
        
        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        x_emb = self.embedder(x)
        encoder_out = self.encoder_stack(x_emb)
        decoder_out = encoder_out
        for decoder_block in self.decoder_stack:
            decoder_out = decoder_block(decoder_out, encoder_out)
        # (seq_len, dModel) -> (seq_len, VOCAB_SIZE)
        linear_proj = self.linear_projection(decoder_out)
        vocab_probabilities = F.softmax(linear_proj, dim=-1)

        return vocab_probabilities

class DecoderOnly(nn.Module):
    # For simiplicity we are using key_dim = value_dim
    def __init__(self, stack_size=6,heads=8, model_dim=512):
        super().__init__()
        assert(model_dim % heads == 0)
        self.embedder = Embedder()
        self.decoder_stack = nn.ModuleList([Decoder() for _ in range(stack_size)])
        self.linear_projection = nn.Linear(512, VOCAB_SIZE)
        # Use custom LayerNorm instead of nn.LayerNorm
        self.final_norm = LayerNorm(model_dim)
        
        # Initialize weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, x):
        # [batch_size, seq_len] -> [batch_size, seq_len, model_dim]
        x_emb = self.embedder(x)

        decoder_out = x_emb
        for decoder_block in self.decoder_stack:
            decoder_out = decoder_block(decoder_out, decoder_out, skip_cross_attn=True)
        
        # Add layer norm before projection
        decoder_out = self.final_norm(decoder_out)
        linear_proj = self.linear_projection(decoder_out)
        return linear_proj

    
    