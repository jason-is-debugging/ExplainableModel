from __future__ import annotations

import os
from typing import List, Tuple

import torch
from torch.utils.data import DataLoader, Dataset
from model.dataset.tokenizer import Tokenizer


class ConversationDataset(Dataset):

    def __init__(
        self, texts: List[str], tokenizer: Tokenizer, max_length: int = 256,
        pad_token_id: int = 0,
    ) -> None:
        super().__init__()

        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = pad_token_id

        self.data = []

        for text in texts:
            tokens = [tokenizer.bos_token_id] + tokenizer.encode(text) + [tokenizer.eos_token_id]
            if len(tokens) > max_length:
                tokens = tokens[:max_length]
            self.data.append(tokens)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> torch.Tensor:
        """Return a padded tensor of shape [max_length]."""
        tokens = self.data[index]
        if len(tokens) > self.max_length:
            tokens = tokens[: self.max_length]
        padding = [self.pad_token_id] * (self.max_length - len(tokens))
        tokens = tokens + padding
        return torch.tensor(tokens, dtype=torch.long)

    def load_text_file(self, file_path: str) -> List[str]:

        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Dataset file not found :{file_path}")

        lines: List[str] = []
        with open(file_path, "r", encoding = "utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]

        return lines

def create_dataloaders(
    train_texts: List[str],
    val_texts: List[str],
    tokenizer: Tokenizer,
    batch_size: int = 32,
    max_length: int = 256,
    num_workers: int = 0,
    pad_token_id: int = 0,
) -> Tuple[DataLoader, DataLoader]:

    train_dataset = ConversationDataset(train_texts, tokenizer, max_length, pad_token_id)
    val_dataset = ConversationDataset(val_texts, tokenizer, max_length, pad_token_id)


    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available())
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers, 
        pin_memory=torch.cuda.is_available())

    return train_loader, val_loader

