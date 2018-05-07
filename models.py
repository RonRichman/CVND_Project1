## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        
        ## Data starts off as (224,224,1); after conv layer it is (220,220,32)
        self.conv1 = nn.Conv2d(1, 32, 5)
        ## Data starts off as (222,222,32); after maxpool layer it is (110,110,32)
        self.pool1 = nn.MaxPool2d(2, 2)
        ##Dropout doesn't affect size of layer
        #self.conv1_drop = nn.Dropout(p=0.25)
        
        ## Data starts off as (110,110,32); after conv layer it is (106,106,64)
        self.conv2 = nn.Conv2d(32, 64, 5)
        ## Data starts off as (106,106,64); after maxpool layer it is (53,53,64)
        self.pool2 = nn.MaxPool2d(2, 2)
        ##Dropout doesn't affect size of layer
        #self.conv2_drop = nn.Dropout(p=0.25)
        self.conv2_bn = nn.BatchNorm2d(64)
        
        ## Data starts off as (53,53,64); after conv layer it is (49,49,128)
        self.conv3 = nn.Conv2d(64, 128, 5)
        ## Data starts off as (49,49,256); after maxpool layer it is (24,24,128)
        self.pool3 = nn.MaxPool2d(2, 2)
        ##Dropout doesn't affect size of layer
        #self.conv2_drop = nn.Dropout(p=0.25)
        
        self.fc1 = nn.Linear(24*24*128, 256)
        self.fc1_bn = nn.BatchNorm1d(256)
        self.fc1_drop = nn.Dropout(p=0.25)
        self.fc2 = nn.Linear(256, 68*2)

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        # a modified x, having gone through all the layers of your model, should be returned
        
        m = nn.LeakyReLU(0.1)
        
        x = self.pool1(m(self.conv1(x)))
        #x = self.conv1_drop(x)
        x = self.pool2(m(self.conv2(x)))
        x = self.conv2_bn(x)
        x = self.pool3(m(self.conv3(x)))
        #x = self.conv2_drop(x)
        
        ## reshape 
        x = x.view(x.size(0), -1)
        x = m(self.fc1(x))
        x = self.fc1_bn(x)
        x = self.fc1_drop(x)
        x = self.fc2(x)
                
        return x
