## TODO: define the convolutional neural network architecture

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# can use the below import should you choose to initialize the weights of your Net
import torch.nn.init as I
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
<<<<<<< HEAD
        if initiation:
=======

            if initiation:
>>>>>>> 56df81caf7cda0ef09a8cfa8c6fca6450362a2af
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

        return x
