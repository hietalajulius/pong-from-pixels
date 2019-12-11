import torch
import random
from collections import namedtuple
import cv2
import numpy as np
import hps
from PIL import Image

def discount_rewards(r, gamma):
    discounted_r = torch.zeros_like(r)
    running_add = 0
    for t in reversed(range(0, r.size(-1))):
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
    return discounted_r

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


def tensor_from_observation(observation, h=hps.HEIGHT, w=hps.WIDTH):
    resized = cv2.resize(observation, (h,w), interpolation = cv2.INTER_LINEAR)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite("../local_files/screenshots/my orig.png", state)
    state = gray_image.reshape((1,1,h,w))
    state = torch.from_numpy(state).float()
    return state

def tensor_from_multiple_observations(observations, h=hps.HEIGHT, w=hps.WIDTH):
    tensors = [tensor_from_observation(observation, h, w) for observation in observations]
    stacked = torch.stack(tensors, dim=1).squeeze(2)
    return stacked

def image_from_observation(observation, h=hps.HEIGHT, w=hps.WIDTH):
    #gray_image = Image.fromarray(observation).resize((width, height), resample=Image.NEARES).convert(mode="L") #PIL
    resized = cv2.resize(observation, (h,w), interpolation = cv2.INTER_LINEAR)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    #cv2.imwrite(str(np.random.choice(range(100)))+'.png', gray_image)
    state = gray_image.reshape((1,1,h,w))
    return state

def stacked_from_multiple_observations(observations, h=hps.HEIGHT, w=hps.WIDTH):
    images = [image_from_observation(observation, h, w) for observation in observations]
    stacked = np.stack(images, 1).squeeze(2)
    return stacked

    

def image_from_observation_ta(observation):
    observation = observation[::2, ::2].mean(axis=-1)
    observation = np.expand_dims(observation, axis=-1)
    ob = observation.reshape((1,1,100,100))
    return ob

def stacked_from_multiple_observations_ta(observations):
    images = [image_from_observation_ta(observation) for observation in observations]
    stacked = np.stack(images, 1).squeeze(2)
    return stacked

def tensor_from_observation_ta(observation):
    observation = observation[::2, ::2].mean(axis=-1)
    observation = np.expand_dims(observation, axis=-1)
    ob = observation.reshape((1,1,100,100))
    state = torch.from_numpy(ob).float()
    return state

def tensor_from_multiple_observations_ta(observations):
    tensors = [tensor_from_observation_ta(observation) for observation in observations]
    stacked = torch.stack(tensors, dim=1).squeeze(2)
    return stacked


def process_2_ta(obs1, obs2):
    obs1 = obs1[::2, ::2].mean(axis=-1)
    obs1 = np.expand_dims(obs1, axis=-1)
    obs2 = obs2[::2, ::2].mean(axis=-1)
    obs2 = np.expand_dims(obs2, axis=-1)

    stack_ob = np.concatenate((obs2, obs1), axis=-1)
    stack_ob = torch.from_numpy(stack_ob).float().unsqueeze(0)
    stack_ob = stack_ob.transpose(1, 3)
    return stack_ob