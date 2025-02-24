import torch
from torch.utils.data import Dataset, DataLoader
from tfrecord.torch.dataset import TFRecordDataset
import os
import itertools
import sentencepiece as spm
import torch.nn.functional as F

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
