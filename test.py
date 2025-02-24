from wikinet_dataset import *  # Import all components from wikinet_dataset.py
from nn_architecture import *  # Import all components from nn_architecture.py
import os
import torch
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch):
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Pad both sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return padded_inputs, padded_targets

# home_path = os.path.expanduser("~")
# # Usage
# tfrecord_path = os.path.join(home_path, "Downloads/wikiweb2m-test.tfrecord")
# index_path = None # Optional for faster indexing
# # dataset = PyTorchTFRecordDataset(tfrecord_path, index_path)

# # For testing, use a smaller subset
# torch.manual_seed(42)  # for reproducibility
# subset_size = 32
# indices = torch.randperm(len(dataset))[:subset_size]
# subset_dataset = torch.utils.data.Subset(dataset, indices)
# dataloader = DataLoader(
#     subset_dataset, 
#     batch_size=32, 
#     shuffle=True,
#     collate_fn=collate_fn  # Add custom collate function
# )

# batch = next(iter(dataloader))
# inputs, targets = batch
# print("[DataLoader] Input batch shape:", inputs.shape)
# print("[DataLoader] Target batch shape:", targets.shape)
# print("[DataLoader] First input item:", inputs[0])
# print("[DataLoader] First target item:", targets[0])

device = torch.accelerator.current_accelerator().type if torch.accelerator.is_available() else "cpu"
# embedder = Embedder().to(device)

# # Generate random integers instead of floats for the embedding layer
# sample = torch.randint(0, 1000, (1000,), device=device)  # Values between 0 and 999
# embed_sample = embedder(sample.T)

# print("[Embedder] Output shape:", embed_sample.shape)

# mha = MultiHeadAttention().to(device)
# mha_sample = mha(embed_sample, embed_sample)
# print("[Multi-Head Attention] Output shape:", mha_sample.shape)

# layer_norm_sample = torch.tensor([1, 2, 3], dtype=torch.float,device=device)
# layer_norm = LayerNorm().to(device)
# layer_norm_output = layer_norm(layer_norm_sample)
# print("[Layer Norm] Output shape:", layer_norm_output.shape)

# encoder_stack = nn.Sequential(*[Encoder().to(device) for _ in range(6)])
# encoder_output = encoder_stack(embed_sample)
# print("[Encoder Stack] Output shape:", encoder_output.shape)

# # Replace the decoder_stack Sequential with a custom stack
# decoder_stack = nn.ModuleList([Decoder().to(device) for _ in range(6)])
# decoder_output = embed_sample
# for decoder in decoder_stack:
#     decoder_output = decoder(decoder_output, embed_sample)
# print("[Decoder Stack] Output shape:", decoder_output.shape)

# embedding output
sequence = torch.zeros(3, dtype=torch.long).to(device)
embedder = Embedder(model_dim=10).to(device)
embed_sample = embedder(sequence)
print(embed_sample)
print("[Embedder] Output shape:", embed_sample.shape)

decoder = Decoder(2,10).to(device)
decoder_output = decoder(embed_sample, embed_sample, skip_cross_attn=True)
print("[Decoder] Output shape:", decoder_output.shape)