import torch
from tfrecord.torch.dataset import TFRecordDataset
import os
import sentencepiece as spm
from tqdm import tqdm
import argparse

def convert_tfrecord_to_pytorch(tfrecord_path, output_dir, batch_size=1000):
    """
    Convert TFRecord dataset to PyTorch tensor files.
    Each batch will be saved as a separate file to manage memory.
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize the TFRecord dataset
    description = {
        "raw_page_description": "byte"
    }
    dataset = TFRecordDataset(tfrecord_path, None, description)
    
    # Initialize SentencePiece tokenizer
    sp = spm.SentencePieceProcessor(model_file="spiece.model")
    
    # Process and save in batches
    batch_inputs = []
    batch_targets = []
    batch_count = 0
    
    print(f"Converting {tfrecord_path}...")
    for idx, example in enumerate(tqdm(dataset)):
        # Process the example
        content_bytes = example["raw_page_description"]
        content_str = content_bytes.decode("utf-8")
        tokens = sp.encode(content_str, out_type=int)
        tensor = torch.tensor(tokens, dtype=torch.long)
        
        # Create input-target pairs (similar to the original dataset)
        inputs = tensor[:-1]
        targets = tensor[1:]
        
        batch_inputs.append(inputs)
        batch_targets.append(targets)
        
        # Save when batch is full or at the end
        if len(batch_inputs) >= batch_size:
            save_batch(batch_inputs, batch_targets, output_dir, batch_count)
            batch_inputs = []
            batch_targets = []
            batch_count += 1
    
    # Save any remaining examples
    if batch_inputs:
        save_batch(batch_inputs, batch_targets, output_dir, batch_count)

def save_batch(inputs, targets, output_dir, batch_num):
    """Save a batch of examples to disk."""
    # Pad sequences in the batch to the same length
    max_len = max(seq.size(0) for seq in inputs)
    padded_inputs = torch.stack([
        torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=0)
        for seq in inputs
    ])
    padded_targets = torch.stack([
        torch.nn.functional.pad(seq, (0, max_len - seq.size(0)), value=0)
        for seq in targets
    ])
    
    # Save the batch
    batch_data = {
        'inputs': padded_inputs,
        'targets': padded_targets,
        'lengths': torch.tensor([len(seq) for seq in inputs])  # Store original lengths
    }
    torch.save(batch_data, os.path.join(output_dir, f'batch_{batch_num}.pt'))

def main():
    parser = argparse.ArgumentParser(description='Convert TFRecord to PyTorch format')
    parser.add_argument('--tfrecord_path', required=True, help='Path to TFRecord file')
    parser.add_argument('--output_dir', required=True, help='Output directory for PyTorch files')
    parser.add_argument('--batch_size', type=int, default=1000, help='Batch size for saving')
    
    args = parser.parse_args()
    convert_tfrecord_to_pytorch(args.tfrecord_path, args.output_dir, args.batch_size)

if __name__ == "__main__":
    main()
