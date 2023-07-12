import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms, datasets 

if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

train_dataset = datasets.MNIST(root ="../data/MNIST",
                               train = True,
                               download = True,
                               transform = transforms.ToTensor( ))

test_dataset = datasets.MNIST(root ="../data/MNIST",
                               train = False,
                            
                               transform = transforms.ToTensor( ))

print(train_dataset,test_dataset)
