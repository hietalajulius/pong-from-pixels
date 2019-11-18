import torch
from torchvision import models
import torch.nn as nn
from torch.distributions.categorical import Categorical

class RnnPolicy(nn.Module):
    def __init__(self):
        super(RnnPolicy, self).__init__()
        self.model = self.initialize_model()
        self.softmax = torch.nn.Softmax()

    def initialize_model(self):
        model_ft = models.resnet18(pretrained=True)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = nn.Linear(num_ftrs, 3)

        return model_ft

    def load_trained_model(self):
        model_filename = 'net_at_100_wins.pth'
        self.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        print('Trained model loaded from %s.' % model_filename)
        self.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.softmax(x)

        dist = Categorical(x)

        return dist
