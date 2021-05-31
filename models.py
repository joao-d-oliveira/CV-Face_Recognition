## TODO: define the convolutional neural network architecture

import torch
<<<<<<< HEAD
from torch.autograd import Variable
=======
>>>>>>> b08f878c1019e6f2383411bbc07c9bf27367cef4
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
<<<<<<< HEAD
from collections import OrderedDict

# Global VARs, SETUP
RANDOM_SEED = 42

class Net_V1_0(nn.Module):

    def __init__(self, initiation=False):
        torch.manual_seed(RANDOM_SEED)
        super(Net_V1_0, self).__init__()

        
        # as seen in an example of paper for NaimishNet, but didn't do the initial dropout with fear would avoid hyper activation of needed features
        self.features = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1, 32, 5)),
              ('relu1', nn.ReLU()),
              ('maxpool1', nn.MaxPool2d(2)),
              ('conv2', nn.Conv2d(32, 64, 5)),
              ('relu2', nn.ReLU()),
              ('maxpool2', nn.MaxPool2d(2)),
              ('dropout1', nn.Dropout(0.4)),
              ('conv3', nn.Conv2d(64, 128, 5)),
              ('relu3', nn.ReLU()),
              ('maxpool3', nn.MaxPool2d(2)),
              ('dropout2', nn.Dropout(0.4)),
              ('conv4', nn.Conv2d(128, 256, 5)),
              ('relu4', nn.ReLU()),
              ('maxpool4', nn.MaxPool2d(2)),
              ('dropout3', nn.Dropout(0.4))
        ]))
        #simplified for now the output
        self.regressor = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(25600, 1000)),
            ('relu1',nn.ReLU()),
            ('dropout1', nn.Dropout(0.4)),
            ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
            ('output', nn.Linear(1000, 136)),
        ]))
        if initiation:
            for layer in chain(self.features.children(), self.regressor.children()):
              if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                  I.xavier_uniform_(layer.weight)
                  I.zeros_(layer.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        
        return x
    
class Net_V1_1(nn.Module):
    def __init__(self, initiation=False):
        torch.manual_seed(RANDOM_SEED)
        super(Net_V1_1, self).__init__()

      # as seen in an example of paper for NaimishNet, but didn't do the initial dropout with fear would avoid hyper activation of needed features
        self.features = nn.Sequential(OrderedDict([
          ('conv1', nn.Conv2d(1, 32, 5)),
          ('relu1', nn.ReLU()),
          ('maxpool1', nn.MaxPool2d(2)),
          ('conv2', nn.Conv2d(32, 64, 5)),
          ('relu2', nn.ReLU()),
          ('maxpool2', nn.MaxPool2d(2)),
          ('dropout1', nn.Dropout(0.4)),
          ('conv3', nn.Conv2d(64, 128, 5)),
          ('relu3', nn.ReLU()),
          ('maxpool3', nn.MaxPool2d(2)),
          ('dropout2', nn.Dropout(0.4)),
          ('conv4', nn.Conv2d(128, 256, 5)),
          ('relu4', nn.ReLU()),
          ('maxpool4', nn.MaxPool2d(2)),
          ('dropout3', nn.Dropout(0.4))
        ]))
      #simplified for now the output
        self.regressor = nn.Sequential(OrderedDict([
          ('linear1',nn.Linear(25600, 12800)),
          ('relu1',nn.ReLU()),
          ('linear2',nn.Linear(12800, 1000)),
          ('relu2',nn.ReLU()),
          ('dropout1', nn.Dropout(0.4)),
          ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
          ('output', nn.Linear(1000, 136)),
        ]))
        
        if initiation:
            for layer in chain(self.features.children(), self.regressor.children()):
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    I.xavier_uniform_(layer.weight)
                    I.zeros_(layer.bias)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)
        return x

class Net_V1_2(nn.Module):
    def __init__(self, initiation=False):
        torch.manual_seed(RANDOM_SEED)
        super(Net_V1_2, self).__init__()
          
          # as seen in an example of paper for NaimishNet, but didn't do the initial dropout with fear would avoid hyper activation of needed features
        self.features = nn.Sequential(OrderedDict([
              ('conv1', nn.Conv2d(1, 32, 5)),
              ('relu1', nn.ReLU()),
              ('maxpool1', nn.MaxPool2d(2)),
              ('conv2', nn.Conv2d(32, 64, 5)),
              ('relu2', nn.ReLU()),
              ('maxpool2', nn.MaxPool2d(2)),
              ('dropout1', nn.Dropout(0.4)),
              ('conv3', nn.Conv2d(64, 128, 5)),
              ('relu3', nn.ReLU()),
              ('maxpool3', nn.MaxPool2d(4)),
              ('dropout2', nn.Dropout(0.4))
            ]))
          #simplified for now the output
        self.regressor = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(128*12*12, 1000)),
            ('relu1',nn.ReLU()),
            ('dropout1', nn.Dropout(0.4)),
              ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
            ('output', nn.Linear(1000, 136)),
           ]))

            if initiation:
            for layer in chain(self.features.children(), self.regressor.children()):
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    I.xavier_uniform_(layer.weight)
                    I.zeros_(layer.bias)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)

        return x

class Net_V1_3(nn.Module):
    def __init__(self, initiation=False):
        torch.manual_seed(RANDOM_SEED)
        super(Net_V1_3, self).__init__()

        # as seen in an example of paper for NaimishNet, but didn't do the initial dropout with fear would avoid hyper activation of needed features
        self.features = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv2d(1, 32, 5)),
            ('relu1', nn.ReLU()),
            ('maxpool1', nn.MaxPool2d(4)),
            ('conv2', nn.Conv2d(32, 64, 5)),
            ('relu2', nn.ReLU()),
            ('maxpool2', nn.MaxPool2d(4)),
            ('dropout1', nn.Dropout(0.4)),
            ]))
            #simplified for now the output
        self.regressor = nn.Sequential(OrderedDict([
            ('linear1',nn.Linear(9216, 1000)),
            ('relu1',nn.ReLU()),
            ('dropout1', nn.Dropout(0.4)),
            ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
            ('output', nn.Linear(1000, 136)),
            ]))

        if initiation:
            for layer in chain(self.features.children(), self.regressor.children()):
                if isinstance(layer, nn.Conv2d) or isinstance(layer, nn.Linear):
                    I.xavier_uniform_(layer.weight)
                    I.zeros_(layer.bias)


    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.regressor(x)

=======


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        ## TODO: Define all the layers of this CNN, the only requirements are:
        ## 1. This network takes in a square (same width and height), grayscale image as input
        ## 2. It ends with a linear layer that represents the keypoints
        ## it's suggested that you make this last layer output 136 values, 2 for each of the 68 keypoint (x, y) pairs
        
        # As an example, you've been given a convolutional layer, which you may (but don't have to) change:
        # 1 input image channel (grayscale), 32 output channels/feature maps, 5x5 square convolution kernel
        self.conv1 = nn.Conv2d(1, 32, 5)
        
        ## Note that among the layers to add, consider including:
        # maxpooling layers, multiple conv layers, fully-connected layers, and other layers (such as dropout or batch normalization) to avoid overfitting
        

        
    def forward(self, x):
        ## TODO: Define the feedforward behavior of this model
        ## x is the input image and, as an example, here you may choose to include a pool/conv step:
        ## x = self.pool(F.relu(self.conv1(x)))
        
        
        # a modified x, having gone through all the layers of your model, should be returned
>>>>>>> b08f878c1019e6f2383411bbc07c9bf27367cef4
        return x
