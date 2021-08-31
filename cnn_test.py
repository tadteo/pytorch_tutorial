#!/env/bin/env python3

import torch
import torch.nn as nn
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt

# 1) Design model (input, output size, forward pass)
# 2) Construct the loss and optimizer
# 3) Training loop:
#       - forward pass: compute prediction
#       - backward pass: calculate gradients
#       - update weights

EPOCHS = 10
BATCH_SIZE=16
LEARNING_RATE = 0.001

# constant for classes
CLASSES = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

class MNIST_Dataset(Dataset):

    def __init___(self):
        pass

class CNN_test(nn.Module):
    """docstring for CNN_test"""
    def __init__(self):
        super(CNN_test, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1 , out_channels=16 , kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(in_channels=16 , out_channels=5 , kernel_size=3, padding=1)
        # self.pool2 = nn.MaxPool2d(2,2)
        # intermediate layers 2 convolution, 2 fully connected ff
        self.fc1 = nn.Linear(5*5*5,64)
        self.fc2 = nn.Linear(64,128)
        #output
        self.fc3 = nn.Linear(128,len(CLASSES))
        print("Neural network created")
    
    def forward(self,x):
        # print(f"Input Size: {x.shape} ")
        x = self.conv1(x)
        # print(f"After conv1 Size: {x.shape}")
        x = self.pool1(x)
        # print(f"After pooling 1 size: {x.shape}")
        x = self.conv2(x)
        # print(f"After conv2 Size: {x.shape}")
        # x = self.pool2(x)
        # print(f"After pooling 2 size: {x.shape}")
        x = torch.flatten(x, 1)
        # print(f"After flatten size: {x.shape}")
        x = self.fc1(x)
        # print(f"After first fully connected size: {x.shape}")
        x = self.fc2(x)
        # print(f"After last layer: {x.shape}")
        x = self.fc3(x)

        return x


def main():
    print("This is just a CNN test")

    #Set gpu if we have cuda cpu otherwise
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transformation = torchvision.transforms.Compose([
        torchvision.transforms.CenterCrop(10),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(0.5, 1)
        ])

    train_dataset = torchvision.datasets.FashionMNIST(root="./data", train = True, download = True, transform=transformation )
    test_dataset = torchvision.datasets.FashionMNIST(root="./data", train = False, download = True, transform=torchvision.transforms.ToTensor())

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=2)

    writer = SummaryWriter('runs/FashionMNIST_experiment_1')

    dataiter = iter(train_loader)
    example_data, example_target = dataiter.next()

    img_grid = torchvision.utils.make_grid(example_data)
    writer.add_image("Fashion MNIST examples", img_grid)
    writer.close()
    
    model = CNN_test().to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Total number of images: {len(train_loader)}")
    for e in range(EPOCHS):
        for i, (images, labels) in enumerate(train_loader):
            # print(f"step: {i} ")
            images.to(device)
            labels.to(device)

            #forward
            # print(f"Images size: {images.shape}")
            output = model(images)
            # print(output)
            loss = criterion(output,labels)

            #backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i%(len(train_loader)/10) == 0:
                print(f"Epoch = {e}, step = {i}/{len(train_loader)} --> {100*i/(len(train_loader))}%, Loss = {loss.item():.4f}")


    print("Training finished!\n\nStarting testing")

    with torch.no_grad():
        n_correct = 0
        for i, (images, labels) in enumerate(test_loader):
            print(i)
            output = model(images)

            test_loss += loss_fn(output, labels).item()
            correct += (output.argmax(1) == y).type(torch.float).sum().item()

            value, index = torch.max(output)
            print(value,index)
            # for i in range(BATCH_SIZE):
                # if labels[i]==index[i]:
                    # n_correct +=1

        accuracy = n_correct/len(test_loader)

        print(f"The accuracy of the network is: {accuracy}")

if __name__ == '__main__':
    main()