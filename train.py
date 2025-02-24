from encoder_decoder import *
from wikinet_dataset import *

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

import os
import time
from collections import defaultdict
from contextlib import contextmanager
import sentencepiece as spm
import torch.nn.functional as F

home_path = os.path.expanduser("~")

# Hyperparameters
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.0001
# VOCAB_SIZE = 100
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Usage
# tfrecord_path = os.path.join(home_path, "Downloads/wiki_web2m_pytorch")

sp = spm.SentencePieceProcessor()
sp.load('spiece.model')

dataset = WikinetDataset(sp_model_path='spiece.model', num_samples=10000)  # Remove tfrecord path as it's no longer needed

# Add after other hyperparameters
WARMUP_STEPS = 4000

def decode_tokens(token_ids):
    # Convert list/tensor of token IDs to text
    if torch.is_tensor(token_ids):
        token_ids = token_ids.cpu().tolist()
    return sp.decode(token_ids)

def collate_fn(batch):
    # Separate inputs and targets
    inputs = [item[0] for item in batch]
    targets = [item[1] for item in batch]
    
    # Pad sequences
    padded_inputs = pad_sequence(inputs, batch_first=True, padding_value=0)
    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0)
    
    return padded_inputs, padded_targets

dataloader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE, 
    shuffle=True,
    collate_fn=collate_fn
)

model = DecoderOnly().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Add after creating optimizer
max_grad_norm = 1.0

# Add after creating optimizer
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer,
    lambda step: min(step / WARMUP_STEPS, 1.0)
)

# Add timing utilities
class Timer:
    def __init__(self):
        self.times = defaultdict(float)
        self.counts = defaultdict(int)
        self.last_report = time.perf_counter()
        
    @contextmanager
    def measure(self, name):
        start = time.perf_counter()
        yield
        self.times[name] += time.perf_counter() - start
        self.counts[name] += 1
    
    def report(self):
        print("\nTiming Report:")
        for name in self.times:
            avg_time = self.times[name] / self.counts[name]
            total_time = self.times[name]
            print(f"{name:20} | Total: {total_time:.2f}s | Avg: {avg_time*1000:.2f}ms | Calls: {self.counts[name]}")
    
    def maybe_report(self, force=False):
        now = time.perf_counter()
        # Report every 5 seconds
        if force or (now - self.last_report) > 5:
            self.report()
            self.last_report = now

# Initialize timer
timer = Timer()

# Add before training loop
timestamp = time.strftime("%Y%m%d_%H%M%S")
checkpoint_dir = os.path.join("checkpoints", f"run_{timestamp}")
os.makedirs(checkpoint_dir, exist_ok=True)

# At the start of training
running_loss = 0
log_every = 10

# Add these at the start of training loop
def debug_batch(batch_idx, inputs, targets, outputs, loss):
    with torch.no_grad():
        print(f"\n=== Debug Batch {batch_idx} ===")
        
        # 1. Check input/target shapes and values
        print(f"Input shape: {inputs.shape}, Target shape: {targets.shape}")
        print(f"Input range: [{inputs.min().item():.2f}, {inputs.max().item():.2f}]")
        print(f"Target range: [{targets.min().item():.2f}, {targets.max().item():.2f}]")
        
        # 2. Check model outputs
        print(f"Output shape: {outputs.shape}")
        print(f"Output range: [{outputs.min().item():.2f}, {outputs.max().item():.2f}]")
        print(f"Output mean: {outputs.mean().item():.2f}, std: {outputs.std().item():.2f}")
        
        # 3. Check for NaNs
        print(f"NaNs in output: {torch.isnan(outputs).any().item()}")
        
        # 4. Look at actual predictions
        logits = outputs.view(-1, VOCAB_SIZE)
        probs = F.softmax(logits, dim=-1)
        print(f"Softmax range: [{probs.min().item():.2e}, {probs.max().item():.2e}]")
        
        # 5. Check gradients after backward pass
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** 0.5
        print(f"Gradient norm: {total_norm:.2f}")
        
        # 6. Sample predictions
        sample_idx = 0
        print("\nSample prediction:")
        print(f"Input tokens: {inputs[sample_idx][:10].tolist()}")
        print(f"Target tokens: {targets[sample_idx][:10].tolist()}")
        pred_tokens = torch.argmax(outputs[sample_idx], dim=-1)
        print(f"Predicted tokens: {pred_tokens[:10].tolist()}")
        
        # 7. Decode some tokens
        print("\nDecoded sample:")
        print(f"Input: {decode_tokens(inputs[sample_idx][:20])}")
        print(f"Target: {decode_tokens(targets[sample_idx][:20])}")
        print(f"Prediction: {decode_tokens(pred_tokens[:20])}")

# Training loop
for epoch in range(EPOCHS):
    model.train()
    total_loss = 0

    epoch_start = time.perf_counter()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        with timer.measure('data_loading'):
            # batch_size, seq_len
            inputs = inputs.to(DEVICE)
            # batch_size, seq_len
            targets = targets.to(DEVICE)

        with timer.measure('forward_pass'):
            optimizer.zero_grad()
            outputs = model(inputs)  # shape: (batch_size, seq_len, vocab_size)
            outputs_flattened = outputs.view(-1, VOCAB_SIZE)
            targets_flattened = targets.view(-1)

            # Calculate loss with masking
            criterion = nn.CrossEntropyLoss(ignore_index=0)
            loss = criterion(outputs_flattened, targets_flattened)
            
            # For logging
            running_loss += loss.item()
            
        with timer.measure('backward_pass'):
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            if batch_idx % log_every == 0:
                print(f"Gradient norm: {grad_norm:.4f}")
            optimizer.step()
            scheduler.step()

        total_loss += loss.item()

        if batch_idx < 3:  # Debug first few batches
            debug_batch(batch_idx, inputs, targets, outputs, loss)

        if batch_idx % log_every == 0:
            with torch.no_grad():
                avg_loss = running_loss / log_every
                running_loss = 0
                print(f"\nBatch {batch_idx}, Average Loss: {avg_loss:.4f}")
                print("Output range:", torch.min(outputs).item(), "to", torch.max(outputs).item())
                
                # Get predictions from the raw logits
                logits = outputs.view(inputs.shape[0], -1, VOCAB_SIZE)
                predictions = torch.argmax(logits, dim=-1)
                print("First few predicted tokens:", predictions[0][:10].tolist())
                print("First few target tokens:", targets[0][:10].tolist())
                print("\nSample from batch:")
                print(f"Input:      {decode_tokens(inputs[0])}")
                print(f"Prediction: {decode_tokens(predictions[0])}")
                print(f"Target:     {decode_tokens(targets[0])}")
            timer.maybe_report()  # Print timing stats every 5 seconds
            print(f"Learning rate: {scheduler.get_last_lr()[0]:.6f}")

    epoch_time = time.perf_counter() - epoch_start
    print(f"Epoch {epoch+1} completed in {epoch_time:.2f}s. Average Loss: {total_loss/len(dataloader):.4f}")
    timer.report()  # Force print timing stats at end of epoch
    
    # Save checkpoint with timestamp
    checkpoint_path = os.path.join(checkpoint_dir, f"model_epoch_{epoch+1}.pt")
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': total_loss/len(dataloader),
    }, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path}")

# Print final timing report
timer.report()

print("Training complete!")
