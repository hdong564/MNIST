import numpy as np 
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms, datasets 

# define 
if torch.cuda.is_available():
    DEVICE = torch.device('cuda')

else:
    DEVICE = torch.device('cpu')

print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

BATCH_SIZE = 32
EPOCHS = 10 


train_dataset = datasets.MNIST(root ="data/MNIST",
                               train = True,
                               download = False,
                               transform = transforms.ToTensor( ))

test_dataset = datasets.MNIST(root ="data/MNIST",
                               train = False,
                            
                               transform = transforms.ToTensor( ))

train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = True)



test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
                                           batch_size = BATCH_SIZE,
                                           shuffle = False)

labels_map = {
    0: "zero",
    1: "one",
    2: "two",
    3: "three",
    4: "four",
    5: "five",
    6: "six",
    7: "seven",
    8: "eight",
    9: "nine",
}

# define MLP model

class NeuralNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(NeuralNet, self).__init__()
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
    
# train

# Generate model
model = NeuralNet(784, 20, 10)  # init(784, 20, 10)
# input dim: 784  / hidden dim: 20  / output dim: 10

# Upload model to GPU
model = model.to('cuda')

# Loss function define (we use cross-entropy)
loss_fn = nn.CrossEntropyLoss()

# Define optimizer
# optimizer = torch.optim.SGD(model.parameters(), lr=0.05) 
optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
# optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

# Train the model
total_step = len(train_loader)

for epoch in range(10):
    for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
        # upload to gpu
        if args.model == "MLP":
            images = images.reshape(-1, 28*28).to('cuda')
        
        
        labels = labels.to('cuda')

        # Forward
        outputs = model(images)  # forwardI(images): get prediction
        loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
        
        # Backward and optimize
        optimizer.zero_grad() #before optimizing, clear to 0
        loss.backward()  # automatic gradient calculation (autograd)
        optimizer.step()  # update model parameter with requires_grad=True 
        
        if (i+1) % 100 == 0:
            print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                   .format(epoch+1, 10, i+1, total_step, loss.item()))
            

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)


with torch.no_grad():   
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.reshape(-1, 28*28).to('cuda') ####question, 
        labels = labels.to('cuda')
        
        outputs = model(images)
        
        _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))