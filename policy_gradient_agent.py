import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from policy_gradient_policies.vgg_policy import VggPolicy
from policy_gradient_policies.cnn_policy import CnnPolicy
from policy_gradient_policies.resnet18_policy import RnnPolicy
import numpy as np
from PIL import Image
import time
import cv2
from utils import discount_rewards


class Agent(object):
    def __init__(self):
        if torch.cuda.is_available():
            print("cuda")
            self.train_device = "cuda"
        else:
            print("cpu")
            self.train_device = "cpu"
        self.policy = RnnPolicy().to(self.train_device)
        self.optimizer = torch.optim.RMSprop(self.policy.parameters(), lr=5e-3)
        self.gamma = 0.99
        self.states = []
        self.action_probs = []
        self.rewards = []


    def get_action(self, observation):
        #resized = cv2.resize(observation, (28,28), interpolation = cv2.INTER_AREA)
        #grayImage = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        #_, blackAndWhiteImage = cv2.threshold(grayImage, 50, 255, cv2.THRESH_BINARY)

        #print("im orig", observation.shape)
        #print("im resized", resized.shape)
        #print("im gray", grayImage.shape)
        #print("im black", blackAndWhiteImage.shape)
        #cv2.imwrite("my orig.png", observation)
        #cv2.imwrite('my res.png', resized)
        #cv2.imwrite('my gr.png', grayImage)
        #cv2.imwrite('my bl.png', blackAndWhiteImage)


        self.states.append(observation)
        #flat = torch.tensor([[blackAndWhiteImage]], dtype=torch.float).to(self.train_device)
        flat = torch.tensor([observation], dtype=torch.float).permute((0,3,1,2)).to(self.train_device)
        dist = self.policy(flat)

        action = dist.sample()
        prob = dist.log_prob(action)

        self.action_probs.append(prob)

        return action

    def load_model(self):
        self.policy.load_trained_model()

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
        gamma_powers = torch.pow(gammas,episode_tensor).to(self.train_device)
        discounted_rewards *= gamma_powers

        # TODO: Compute the optimization term (T1)
        loss = -torch.mean(discounted_rewards * action_probs)

        # TODO: Compute the gradients of loss w.r.t. network parameters (T1)
        loss.backward()
        # TODO: Update network parameters using self.optimizer and zero gradients (T1)
        self.optimizer.step()
        self.optimizer.zero_grad()

