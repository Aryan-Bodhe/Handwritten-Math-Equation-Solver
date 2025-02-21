import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import torch
import torch.nn as nn
import torch.nn.functional as F

from model.dataloader import train_dataset_y

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Define CNN Model
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3) # 1 input channel, 32 output channels, 3x3 kernel
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3)
        self.relu3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.fc1 = nn.Linear(256*2*2, 512)  
        self.relu4 = nn.ReLU()
        self.fc2 = nn.Linear(256, len(train_dataset_y)) 
        
    def forward(self, x):
        x = self.pool1(self.relu1(self.conv1(x)))
        print("After Conv1:", x.shape)
        x = self.pool2(self.relu2(self.conv2(x)))
        print("After Conv2:", x.shape)
        x = self.pool3(self.relu3(self.conv3(x)))
        print("After Conv3:", x.shape)
        x = x.view(x.shape[0], -1)    
        print("After Reshape:", x.shape)
        x = self.relu4(self.fc1(x))
        print("After FC1:", x.shape)
        x = self.fc2(x) 
        print("After FC2:", x.shape)
        return x
