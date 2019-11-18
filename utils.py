import torch
import random
from collections import namedtuple
import cv2
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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


def tensor_from_observation(observation):
    resized = cv2.resize(observation, (84,84))# interpolation = cv2.INTER_AREA)
    gray_image = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    _, black_and_white_image = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

    '''
    print("im orig", state.shape)
    print("im resized", resized.shape)
    print("im gray", grayImage.shape)
    print("im black", blackAndWhiteImage.shape)
    cv2.imwrite("./screenshots/my orig.png", state)
    cv2.imwrite('./screenshots/my res.png', resized)
    cv2.imwrite('./screenshots/my gr.png', gray_image)
    
    rand = np.random.uniform()
    if rand > 0.95:
        cv2.imwrite('./screenshots/my_bl +'+ str(rand) +'.png', black_and_white_image)
    '''
    

    state = black_and_white_image.reshape((1,1,84,84))
    state = torch.from_numpy(state).float()
    return state

def tensor_from_multiple_observations(observations):
    if len(observations) != 4:
        print("obs", observations)
        print("Incorrent amount of states provided")
    tensors = [tensor_from_observation(observation) for observation in observations]
    stacked = torch.stack(tensors, dim=1).squeeze(2)
    return stacked.to(device)