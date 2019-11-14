import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import time


def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r


class Policy(torch.nn.Module):
    def __init__(self, state_space, action_space):
        super().__init__()
        self.state_space = state_space
        self.action_space = action_space
        self.hidden = 64

        self.conv1 = nn.Conv2d(3, 16, 10, padding=0)
        self.conv2 = nn.Conv2d(16, 32, 10, padding=0)
        self.fc1 = nn.Linear((43 * 43 *32), 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 3)

        self.softmax = torch.nn.Softmax()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if type(m) is torch.nn.Linear:
                torch.nn.init.normal_(m.weight)
                torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = x.view(-1, (32 * 43 * 43))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        xn = self.fc3(x)
        x = self.softmax(xn)
        return x, xn


class Agent(object):
    def __init__(self):
        self.train_device = "cpu"
        self.policy = Policy(200*200*3, 3).to(self.train_device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=5e-3)
        self.gamma = 0.98
        self.states = []
        self.action_probs = []
        self.rewards = []


    def get_action(self, observation):
        self.states.append(observation)
        flat = torch.tensor([observation], dtype=torch.float).permute((0,3,1,2))
        action, non_soft = self.policy(flat)

        self.action_probs.append(torch.tensor([np.amax(action.detach().numpy())], requires_grad=True))
        print("action", action, "non soft", non_soft)
        return np.argmax(action.detach().numpy())

    def get_name(self):
        return "Better than you"

    def store_reward(self, reward): #Only needed during training
        self.rewards.append(torch.tensor([reward], dtype=torch.float))

    def update_network(self):
        action_probs = torch.stack(self.action_probs, dim=0) \
        .to(self.train_device).squeeze(-1)
        rewards = torch.stack(self.rewards, dim=0).to(self.train_device).squeeze(-1)
        self.states, self.action_probs, self.rewards = [], [], []

        # TODO: Compute discounted rewards (use the discount_rewards function)
        discounted_rewards = discount_rewards(rewards, self.gamma)

        discounted_rewards -= torch.mean(discounted_rewards) # only for Task 1 (c) and keep for Task 2
        discounted_rewards /= torch.std(discounted_rewards) # only for Task 1 (c) and keep for Task 2

        episode_tensor = torch.tensor(range(len(discounted_rewards)), dtype=torch.float)
        gammas = torch.zeros_like(episode_tensor) + torch.tensor([self.gamma], dtype=torch.float)
        gamma_powers = torch.pow(gammas,episode_tensor)
        discounted_rewards *= gamma_powers

        # TODO: Compute the optimization term (T1)
        loss = -torch.mean(discounted_rewards * action_probs)

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

