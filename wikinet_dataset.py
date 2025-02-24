import torch
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset
import os
import itertools
import sentencepiece as spm
import torch.nn.functional as F
from datasets import load_dataset

home_path = os.path.expanduser("~")


class ConvertedWikiDataset(torch.utils.data.Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.batch_files = sorted([f for f in os.listdir(data_dir) if f.endswith('.pt')])
        
        # Calculate total length
        self.total_samples = 0
        self.batch_starts = []
        for batch_file in self.batch_files:
            batch_data = torch.load(os.path.join(data_dir, batch_file))
            self.total_samples += len(batch_data['inputs'])
            self.batch_starts.append(self.total_samples)
        self.current_batch = None
        self.current_batch_idx = None
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        batch_idx = next(i for i, start in enumerate(self.batch_starts) if start > idx)
        
        # Load batch only if different from current
        if self.current_batch_idx != batch_idx:
            self.current_batch = torch.load(os.path.join(self.data_dir, self.batch_files[batch_idx]))
            self.current_batch_idx = batch_idx
        
        prev_start = 0 if batch_idx == 0 else self.batch_starts[batch_idx - 1]
        within_batch_idx = idx - prev_start
        length = self.current_batch['lengths'][within_batch_idx]
        
        return (
            self.current_batch['inputs'][within_batch_idx][:length],
            self.current_batch['targets'][within_batch_idx][:length]
        )


class WikinetDataset(torch.utils.data.Dataset):
    def __init__(self, sp_model_path, max_length=512, split="train", num_samples=None):
        # Stream Wikipedia dataset from HuggingFace
        self.dataset = load_dataset(
            "wikipedia", 
            "20220301.en", 
            split=split, 
            streaming=True
        )
        
        if num_samples is not None:
            self.dataset = self.dataset.take(num_samples)
        
        # Initialize SentencePiece tokenizer
        self.sp = spm.SentencePieceProcessor()
        self.sp.Load(sp_model_path)
        self.max_length = max_length
        
        # Convert streaming dataset to list for length calculation and indexing
        self.data = list(itertools.islice(self.dataset, num_samples if num_samples else None))
        self.total_samples = len(self.data)
    
    def __len__(self):
        return self.total_samples
    
    def __getitem__(self, idx):
        # Get text from cached data
        text = self.data[idx]["text"]
        
        # Tokenize the text using SentencePiece
        tokens = self.sp.EncodeAsIds(text)
        
        # Truncate if necessary
        if len(tokens) > self.max_length:
            tokens = tokens[:self.max_length]
        
        # Convert to tensor
        tokens = torch.tensor(tokens, dtype=torch.long)
        
        # Create inputs and targets (shifted by 1)
        inputs = tokens[:-1]
        targets = tokens[1:]
        
        # Get actual sequence length
        length = len(inputs)
        
        return inputs, targets
