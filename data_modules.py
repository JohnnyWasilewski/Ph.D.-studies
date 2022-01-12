from PIL import Image

import numpy as np

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torch.utils.data import random_split
from sklearn.model_selection import train_test_split
import torch
from torchvision import transforms


class MyDatasetOfImages(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, idx):
        x = self.data[idx]
        x = torch.tensor(np.expand_dims(x, 0), dtype=torch.float)
        y = self.targets[idx]
        return x, y
    
    def __len__(self):
        return len(self.targets)
    
class MyDataset(Dataset):
    def __init__(self, data, targets, transform):
        self.data = data
        self.targets = targets
        self.transform = transform
        
    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]
        return x, y
    
    def __len__(self):
        return len(self.targets)
    
class MyDataModule(pl.LightningDataModule):
    def __init__(self, X, y, transform, num_workers, dataset):
        super().__init__()
        self.X = X
        self.y = torch.FloatTensor(y).to(torch.float64)
        self.transform = transform
        self.num_workers = num_workers
        self.num_classes = np.unique(y)
        self.dims = X[0].shape
        self.dataset = dataset
    
    def setup(self, stage):
        X_train_val, X_test, y_train_val, y_test = train_test_split(self.X, self.y)
        X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val)
        self.train_dataset = self.dataset(X_train, y_train, self.transform)
        self.test_dataset = self.dataset(X_test, y_test, self.transform)
        self.val_dataset = self.dataset(X_val, y_val, self.transform)
    
    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=50, num_workers=self.num_workers)
    
    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=50, num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=50, num_workers=self.num_workers)


class MNISTDataModule(pl.LightningDataModule):
    def __init__(self, data_dir='./', batch_size=32):

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_classes = 10
        self.dims = (1, 28, 28)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(100)
            #transforms.Normalize((0.1307,), (0.3081,))
        ])

    def prepare_data(self):
        # download
        MNIST(self.data_dir, train=True, download=True)
        MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        if stage == 'fit' or stage is None:
            mnist_full = MNIST(self.data_dir, train=True, transform=self.transform,
                               target_transform=lambda x: float(x<5))
            self.mnist_train, self.mnist_val = random_split(mnist_full, [55000, 5000])

        if stage == 'test' or stage is None:
            self.mnist_test = MNIST(self.data_dir, train=False, transform=self.transform,
                                   target_transform=lambda x: float(x<5))

    def train_dataloader(self):
        return DataLoader(self.mnist_train, batch_size=self.batch_size)

    def val_dataloader(self):
        return DataLoader(self.mnist_val, batch_size=self.batch_size)

    def test_dataloader(self):
        return DataLoader(self.mnist_test, batch_size=self.batch_size)