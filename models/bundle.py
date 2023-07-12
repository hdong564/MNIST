
import torch.nn as nn 
import torch.nn.functional as F
import torch


# MLP
class mlpNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(mlpNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size) 
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.sigmoid = nn.Sigmoid()  # sigmoid activation function (you can customize)
    
    def forward(self, x):
        out = self.fc1(x)
        out = self.sigmoid(out)
        out = self.fc2(out)
        out = self.sigmoid(out)
        out = self.fc3(out)
        return out
    
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 32 output channels, 7x7 square convolution, 1 stride
        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 32, 7),
            nn.BatchNorm2d(32),
            nn.ReLU(),
        )
        # 32 input image channel, 64 output channels, 7x7 square convolution, 1 stride
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, 7),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.fc = nn.Linear(64 * 16 * 16, 10)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)
        return out