import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor


class CustomDataset(Dataset):
    def __init__(self, custom_images):
        self.custom_images = custom_images
        self.transform = ToTensor()

    def __len__(self):
        return len(self.custom_images)

    def __getitem__(self, idx):
        image = self.custom_images[idx]
        image = self.transform(image)
        return image

