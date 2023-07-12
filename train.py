import sys
import time
import os
import argparse

import glob
import datetime
from models.bundle import *
from custom_dataset import *
import numpy as np 

import matplotlib
import matplotlib.pyplot as plt

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torchvision import transforms, datasets 
from torch.utils.data import Dataset, DataLoader

from PIL import Image


parser = argparse.ArgumentParser(description = "MNIST Training");

## Data loader
parser.add_argument('--batch_size',         type=int, default=50,	help='Batch size, number of classes per batch');

## Training details
parser.add_argument('--test_interval',  type=int,   default=5,      help='Test and save every [test_interval] epochs');
parser.add_argument('--max_epoch',      type=int,   default=100,    help='Maximum number of epochs');
parser.add_argument('--model',      type=str,   default="MLP",    help='Choose model for training');
## Load and save
parser.add_argument('--initial_model',  type=str,   default="",     help='Initial model weights');
parser.add_argument('--save_path',      type=str,   default="model_params", help='Path for model and logs');
parser.add_argument('--custom', type = int, default = 0, help = 'handwritten dataset dir')
parser.add_argument('--eval', type = int, default = 0, help = '0 for trining, 1 for evaluation')
parser.add_argument('--init_model', type = str, default = "model_params/CNN_95.pth", help = 'load classification model')

args = parser.parse_args();


def train(args):

    ## put on cuda
    if torch.cuda.is_available():
        DEVICE = torch.device('cuda')

    else:       
        DEVICE = torch.device('cpu')

    print('Using PyTorch version:', torch.__version__, ' Device:', DEVICE)

    #params
    BATCH_SIZE = args.batch_size
    EPOCHS = args.max_epoch


    ## load dataset
    dataset = ""
    if args.custom == 0:
        print("Load MNIST dataset")
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
    else: # use custom dataset
        print("load custom dataset")
        dataset = NumDataset(path = './custom_data', train = False, transform = transforms.ToTensor())
        custom_dataloader = torch.utils.data.DataLoader(dataset = dataset,
                                                  batch_size = 1,
                                                  shuffle = True, 
                                                  drop_last = False)

        model = torch.load(args.init_model)
        print("load model:{}".format(args.init_model) )
        
        model.eval()
        with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in custom_dataloader:
                    images, labels = images.to('cuda') ,labels.to('cuda')
                    if args.model == "MLP":
                        images = images.reshape(-1, 28*28).to('cuda')
                    print(images.size(), labels)
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of custom digit dataset: {} %'.format(100 * correct / total))
        sys.exit()

    ## load models
    model = ""

    if args.model == "MLP":
        model = mlpNet(784,20,10)
    elif args.model == "CNN":
        model = CNN()
    
    #upload model to GPU
    model = model.to('cuda')

    #loss function define
    loss_fn = nn.CrossEntropyLoss()

    # Define optimizer
    # optimizer = torch.optim.SGD(model.parameters(), lr=0.05) 
    optimizer = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    # optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # Train the model
    total_step = len(train_loader)
    test_interval = args.test_interval

    # write scorefile
    scorefile_name = args.save_path + '/scores_' + '{}'.format(args.model) + '.txt'
    scorefile = open(scorefile_name,"w")
    strtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    scorefile.write('{}\n{}\n'.format(strtime,args))
    scorefile.flush()
    
    losses = []
    epochs = 0
    step = 0

    test_accuracy = []


    for epoch in range (EPOCHS):
        
        train_loss = 0.0

        for i, (images, labels) in enumerate(train_loader):  # mini batch for loop
            # upload to gpu
            if args.model == "MLP":
                images = images.reshape(-1, 28*28).to('cuda')

            
            images, labels = images.to('cuda'), labels.to('cuda')
            # Forward
            outputs = model(images)  # forwardI(images): get prediction
            loss = loss_fn(outputs, labels)  # calculate the loss (crossentropy loss) with ground truth & prediction value
            
            # Backward and optimize
            optimizer.zero_grad() #before optimizing, clear to 0
            loss.backward()  # automatic gradient calculation (autograd)
            optimizer.step()  # update model parameter with requires_grad=True 
            
            if (i+1) % 100 == 0:
                print ('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}' 
                    .format(epoch+1, EPOCHS, i+1, total_step, loss.item()))

            train_loss += loss.item()
            scorefile.write("EPOCH {:d}, LOSS {:.5f}\n".format(epoch, loss.item()));
        
        # for visualizing loss per epoch
        losses.append(train_loss)
        epochs +=1

        # evaluate the model every test_interval
        if epoch % test_interval == 0:
            with torch.no_grad():
                correct = 0
                total = 0
                for images, labels in test_loader:
                    if args.model == "MLP":
                        images = images.reshape(-1, 28*28).to('cuda')

                    images, labels = images.to('cuda') ,labels.to('cuda')
                    
                    outputs = model(images)
                    
                    _, predicted = torch.max(outputs.data, 1)  # classificatoin model -> get the label prediction of top 1 
                    total += labels.size(0)
                    correct += (predicted == labels).sum().item()

                print('Accuracy of the network on the 10000 test images: {} %'.format(100 * correct / total))
                scorefile.write("EPOCH {:d}, Accuracy {:.5f}\n".format(epoch, 100*correct/total))
                test_accuracy.append(100*correct/total)
                # save model
                torch.save(model,args.save_path +'/' + args.model + '_{}.pth'.format(epoch))    

    
    #visualize performance
    plt.subplot(2,1,1)
    plt.plot(losses)
    plt.xlabel('epoch')
    plt.ylabel('total loss per epoch')
    plt.title('loss graph')

    plt.subplot(2,1,2)
    epoch_x = np.linspace(0, 500, 100,dtype = int)

    plt.plot(epoch_x, test_accuracy, 'o-')
    plt.title('Accuracy graph')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')

    plt.tight_layout()
    plt.show()
        
def main():
    if not(os.path.exists(args.save_path)):
        os.makedirs(args.save_path)
    train(args)


if __name__ == '__main__':
    main()