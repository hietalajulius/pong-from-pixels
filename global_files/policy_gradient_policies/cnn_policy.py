import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical

class CnnPolicy(torch.nn.Module):
    def __init__(self):
        super(CnnPolicy, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, (8,8), stride=4)
        self.conv2 = nn.Conv2d(32, 64, (4,4), stride=2)
        self.conv3 = nn.Conv2d(64, 64, (3,3), stride=1)
        self.fc1 = nn.Linear(28224,512)
        self.fc_action = nn.Linear(512,3) 
        self.softmax = torch.nn.Softmax()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        x = x.view(-1, 28224)

        x = F.relu(self.fc1(x))
        x = F.relu(self.fc_action(x))
        x = self.softmax(x)
        dist = Categorical(x)

        return dist