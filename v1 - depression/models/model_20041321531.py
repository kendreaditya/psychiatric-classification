import torch.nn as nn
import torch
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 64, 3, 1)
        self.conv2 = nn.Conv2d(64, 128, 3, 1)
        self.conv3 = nn.Conv2d(128, 256, 3, 1)

        self.fc_input = 256
        
        self.fc1 = nn.Linear(self.fc_input, 5120)
        self.fc2 = nn.Linear(5120, 512)
        self.fc3 = nn.Linear(512, 2)

    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (3,3))
        x = F.max_pool2d(F.relu(self.conv2(x)), (3,3))
        x = F.max_pool2d(F.relu(self.conv3(x)), (3,3))
        #x = torch.flatten(x)
        #print(x.shape)
        x = x.view(-1, self.fc_input)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
