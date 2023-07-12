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

# visualizing data

figure = plt.figure(figsize =(8,8))
cols, rows = 3,3


# length of training dataset is 60000
# length of test dataet is 10000


for i in range(1, cols*rows+1):
    sample_idx = torch.randint(len(train_dataset), size=(1,)).item()
    img, label = train_dataset[sample_idx]
    print(type(label))
    print(img.size())
    figure.add_subplot(rows,cols,i)
    plt.title(labels_map[label])
    plt.axis('off')
    plt.imshow(img.squeeze(),cmap = "gray")
plt.show()

# for epoch in range(2):
#     print(f"epoch : {epoch} ")
#         for batch in dataloader:
#             img, label = batch
            # print(img.size(), label[0])