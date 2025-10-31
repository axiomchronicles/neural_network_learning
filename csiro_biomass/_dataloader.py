import torch
import torchvision
import pandas as pd
import typing

from torchvision import transforms as T
from torch.utils import data

RANDOM_SEED = 42

class TrainingDatasetLoader:
    def __init__(self, dataset: data.Dataset, batch_size: int | None= 10, shuffle: bool | None = True,
                 pin_memory: bool | None = False, num_workers: int | None = 0, drop_last: bool | None = False):
        
        self.dataset: data.Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last

    def _split_dataset(self):
        torch.manual_seed(RANDOM_SEED)
        train_size = int(0.8 * len(self.dataset))
        val_size = len(self.dataset) - train_size

        X, y = data.random_split(dataset = self.dataset, lengths = [train_size, val_size])
        return X, y
    
    def load_trainingset(self):
        X, _ = self._split_dataset()
        _loader = data.DataLoader(dataset = X, batch_size = self.batch_size,
                shuffle = self.shuffle, pin_memory = self.pin_memory,
                num_workers = self.num_workers, drop_last = self.drop_last)
        
        return _loader
    
    def load_validationset(self):
        _, y = self._split_dataset()
        _loader = data.DataLoader(dataset = y, batch_size = self.batch_size,
                shuffle = self.shuffle, pin_memory = self.pin_memory,
                num_workers = self.num_workers, drop_last = self.drop_last)
        return _loader
    
class TestingDatasetLoader:
    def __init__(self, dataset: data.Dataset, batch_size: int | None= 10, shuffle: bool | None = True,
                 pin_memory: bool | None = False, num_workers: int | None = 0, drop_last: bool | None = False):
        
        self.dataset: data.Dataset = dataset
        self.batch_size: int = batch_size
        self.shuffle: bool = shuffle
        self.pin_memory: bool = pin_memory
        self.num_workers: int = num_workers
        self.drop_last: bool = drop_last

    def load_testingset(self):
        _loader = data.DataLoader(dataset = self.dataset, batch_size = self.batch_size,
                shuffle = self.shuffle, pin_memory = self.pin_memory,
                num_workers = self.num_workers, drop_last = self.drop_last)
        
        return _loader