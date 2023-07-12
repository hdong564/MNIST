# MNIST
This repository is for MNIST classification model




# Training log

## MNIST dataset

### CNN implementation
```
python3 train.py --model CNN
```
- best acurracy model: 'CNN_95.pth' (accuracy 98.77000)
- EPOCH 95, Accuracy 98.77000

### MLP implementation

```
python3 train.py -- model MLP
```
- best acurracy model: 'MLP_25.pth'
- EPOCH 25, Accuracy 96.14000


#### With Custom data

##### CNN95
```
python3 train.py --custom 1 --model CNN --init_model model_params/CNN_95.pth
```

- result: 
Accuracy of custom digit dataset: 6.666666666666667 % (black letter and white background)
Accuracy of custom digit dataset: 66.66666666666667 % (white letter and black background)

##### MLP25

```
python3 train.py --custom 1 --init_model model_params/MLP_25.pth --model MLP
```
- result
Accuracy of custom digit dataset: 3.3333333333333335 % (black letter and white background)
Accuracy of custom digit dataset: 43.333333333333336 % (white letter and black background)
## Image recognition

- ref: https://jeo96.tistory.com/entry/CIFAR-10-CNN%EC%9C%BC%EB%A1%9C-%ED%95%99%EC%8A%B5%ED%95%98%EA%B8%B0-Pytorch
``` ruby
BATCH_SIZE = 32
train_dataset = datasets.CIFAR10(root="./data/",
                                 train=True,
                                 download=True,
                                 transform=transforms.ToTensor())

test_dataset = datasets.CIFAR10(root="./data/",
                                train=False,
                                download=True,
                                transform=transforms.ToTensor())

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=BATCH_SIZE,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=BATCH_SIZE,
                                          shuffle=False)

print(train_loader.dataset)
# Dataset CIFAR10
#    Number of datapoints: 50000
#    Root location: ./data/
#    Split: Train
#    StandardTransform
#    Transform: ToTensor()
```

```  ruby
#CNN model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3,
            out_channels=8,
            kernel_size=3,
            padding=1)
        self.conv2 = nn.Conv2d(
            in_channels=8,
            out_channels=16,
            kernel_size=3,
            padding=1)
        self.pool = nn.MaxPool2d(
            kernel_size=2,
            stride=2
        )
        self.fc1 = nn.Linear(8 * 8 * 16, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = torch.relu(x)
        x = self.pool(x)

        x = x.view(-1, 8 * 8 * 16)
        x = self.fc1(x)
        x = torch.relu(x)
        x = self.fc2(x)
        x = torch.relu(x)
        x = self.fc3(x)
        x = torch.log_softmax(x, dim=1)
        return x

```

## Comparison of CNN and MLP in efficiency in digit recognition

    MLP (Multi-Layer Perceptron):
    MLPs are a type of feedforward artificial neural network that consists of at least three layers of nodes: an input layer, a hidden layer, and an output layer. Each node in one layer is connected to every node in the next layer, making it fully connected or "dense". While MLPs can be used effectively for many simple pattern recognition tasks, they have some limitations when it comes to more complex tasks like image recognition:

    MLPs treat input data as a flat vector, which means that they don't consider the 2D spatial structure of an image. In the case of digit recognition, this means that the MLP may struggle to recognize digits that are shifted or rotated because it doesn't understand the spatial relationships between pixels.

    MLPs are fully connected networks. In case of large images, this results in a large number of parameters, which increases the complexity and computational demands of the network.

    CNN (Convolutional Neural Network):
    CNNs are a type of neural network specifically designed to process pixel data and are particularly effective for image recognition tasks, including digit recognition. Here's why:

    CNNs maintain the spatial structure of the image, as they take a 2D input. This allows them to recognize spatial hierarchies in a way that MLPs can't. For example, a CNN can learn to recognize edges, then use combinations of edges to recognize more complex shapes, and so on.

    CNNs share weights across space. This property drastically reduces the number of parameters, making the network less complex and more computationally efficient compared to MLPs. This makes them more efficient to train, less prone to overfitting, and better at generalizing to new examples.

    The architecture of a CNN also includes pooling layers, which reduce the dimensionality of the data while maintaining the most important features, providing a kind of built-in feature selection.

    So, while both MLPs and CNNs can be used for digit recognition tasks, CNNs are generally more efficient and achieve higher performance on this kind of task. CNNs take into account the spatial structure of images, which is crucial for recognizing digits, as the arrangement of pixels is important in defining what a digit is.

## Comparison of CNN and MLP in terms of efficiency in general image recognition

    CNNs (Convolutional Neural Networks) are generally more efficient and effective than MLPs (Multi-Layer Perceptrons) for general image recognition tasks due to the following reasons:

    Spatial Hierarchies of Patterns: CNNs are able to understand the spatial hierarchies in an image by maintaining the spatial structure of the input. For example, a CNN can learn to recognize simple patterns, such as edges, then use these to recognize more complex patterns, such as shapes, and further use these to recognize even more complex patterns, and so on. This isn't something that MLPs, which treat the input as a flat vector and don't understand the spatial structure, can do effectively.

    Parameter Efficiency: The weight sharing feature of CNNs drastically reduces the number of parameters in the model. This leads to less computational requirements, more efficient training, and lower risk of overfitting. In contrast, MLPs are fully connected networks, which means the number of parameters can increase dramatically with each additional layer or neuron, leading to higher computational demands and a higher risk of overfitting.

    Translation Invariance: CNNs, due to their architecture and the use of pooling layers, are able to maintain translational invariance, meaning they can recognize objects irrespective of where they are located in the image. This is a very important feature for image recognition tasks. MLPs, on the other hand, treat each input separately and hence, do not maintain this property.

    Dimensionality Reduction: The pooling layers in CNNs help to reduce the spatial size (dimensionality) of the data, thus further reducing computational requirements while still maintaining the most important features.

    Therefore, for general image recognition tasks, CNNs are typically much more efficient and effective than MLPs. However, there are tasks for which MLPs could be more suited, such as when the input data is one-dimensional and the spatial structure is not important.
