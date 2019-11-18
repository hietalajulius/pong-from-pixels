import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class VggPolicy(torch.nn.Module):
    def __init__(self):
        super(VggPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, 3,padding=[1,1])
        self.bn1 = nn.BatchNorm2d(num_features=16)
        
        self.conv2 = nn.Conv2d(16, 16, 3,padding=[1,1])
        self.bn2 = nn.BatchNorm2d(num_features=16)
        
        self.conv3 = nn.Conv2d(16, 16, 3,padding=[1,1])
        self.bn3 = nn.BatchNorm2d(num_features=16)
        
        self.conv4 = nn.Conv2d(16, 32, 3,padding=[1,1])
        self.bn4 = nn.BatchNorm2d(num_features=32)
        
        self.conv5 = nn.Conv2d(32, 32, 3,padding=[1,1])
        self.bn5 = nn.BatchNorm2d(num_features=32)
        
        self.conv6 = nn.Conv2d(32, 32, 3,padding=[1,1])
        self.bn6 = nn.BatchNorm2d(num_features=32)
        
        self.conv7 = nn.Conv2d(32, 48, 3,padding=0)
        self.bn7 = nn.BatchNorm2d(num_features=48)
        
        self.conv8 = nn.Conv2d(48, 32, 1,padding=0)
        self.bn8 = nn.BatchNorm2d(num_features=32)
        
        self.conv9 = nn.Conv2d(32, 16, 1,padding=0)
        self.bn9 = nn.BatchNorm2d(num_features=16)
        
        self.fc1 = nn.Linear(16,3)

        self.softmax = torch.nn.Softmax()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)
        
        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)
        
        x = self.conv3(x)
        x = self.bn3(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        
        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)
        
        x = self.conv5(x)
        x = self.bn5(x)
        x = F.relu(x)
        
        x = self.conv6(x)
        x = self.bn6(x)
        x = F.max_pool2d(F.relu(x), (2, 2))
        
        x = self.conv7(x)
        x = self.bn7(x)
        x = F.relu(x)
        
        x = self.conv8(x)
        x = self.bn8(x)
        x = F.relu(x)
        
        x = self.conv9(x)
        x = self.bn9(x)
        x = F.relu(x)
        
        x = F.avg_pool2d(x,5)
        x = x.view(-1,16)
        
        x = self.fc1(x)
    
        x = self.softmax(x)
        dist = Categorical(x)

        return dist