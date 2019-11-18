import torch
import torch.nn as nn
from utils import ReplayMemory, Transition, tensor_from_multiple_observations
import torch.optim as optim
import torch.nn.functional as F
import random
import numpy as np
import cv2

class DQN(nn.Module):
    def __init__(self, h=84, w=84, outputs=3, input_channels=4):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(input_channels, 16, kernel_size=5, stride=2)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, stride=2)
        self.bn2 = nn.BatchNorm2d(32)
        self.conv3 = nn.Conv2d(32, 32, kernel_size=5, stride=2)
        self.bn3 = nn.BatchNorm2d(32)

        # Number of Linear input connections depends on output of conv2d layers
        # and therefore the input image size, so compute it.
        def conv2d_size_out(size, kernel_size = 5, stride = 2):
            return (size - (kernel_size - 1) - 1) // stride  + 1
        convw = conv2d_size_out(conv2d_size_out(conv2d_size_out(w)))
        convh = conv2d_size_out(conv2d_size_out(conv2d_size_out(h)))
        linear_input_size = convw * convh * 32
        self.head = nn.Linear(linear_input_size, outputs)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        return self.head(x.view(x.size(0), -1))

    def load_trained_model(self):
        model_filename = 'net_at_12876_games.pth'
        self.load_state_dict(torch.load(model_filename, map_location=lambda storage, loc: storage))
        print('Trained model loaded from %s.' % model_filename)
        self.eval()


class Agent(object):
    def __init__(self, replay_buffer_size=100000,
                 batch_size=32, gamma=0.999):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.n_actions = 3
        self.policy_net = DQN().to(self.device)
        self.target_net = DQN().to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        self.optimizer = optim.RMSprop(self.policy_net.parameters(), lr=1e-3)
        self.memory = ReplayMemory(replay_buffer_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.observation_history = [np.zeros((200,200,3)).astype(np.float32) for _ in range(4)]
        self.new_state = tensor_from_multiple_observations(self.observation_history[-4:])
        self.previous_state = tensor_from_multiple_observations(self.observation_history[-4:])
        self.epsilon = 0.1
        self.frames_seen = 0
        self.target_epsilon = 0.1
        self.target_frames = 10000000
        self.a = int((self.target_epsilon*self.target_frames)/(1-self.target_epsilon))
        self.train = True

    def get_name(self):
        return "Better than you"
    
    def load_model(self):
        self.policy_net.load_trained_model()

    def reset(self):
        self.observation_history = [np.zeros((200,200,3)).astype(np.float32) for _ in range(4)]
        self.new_state = tensor_from_multiple_observations(self.observation_history[-4:])
        self.previous_state = tensor_from_multiple_observations(self.observation_history[-4:])

    def update_network(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = 1-torch.tensor(batch.done, dtype=torch.uint8, device=self.device)
        non_final_next_states = [s for nonfinal,s in zip(non_final_mask,
                                     batch.next_state) if nonfinal > 0]
        non_final_next_states = torch.stack(non_final_next_states).squeeze(1)

        state_batch = torch.stack(batch.state).squeeze(1)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=self.device)
        next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1)[0].detach()
        # Task 4: TODO: Compute the expected Q values
        expected_state_action_values = self.gamma*next_state_values + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values.squeeze(),
                                expected_state_action_values)

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.policy_net.parameters():
            param.grad.data.clamp_(-1e-1, 1e-1)
        self.optimizer.step()

    def get_action(self, observation, epsilon=0.05):
        self.frames_seen += 1
        if self.frames_seen < self.target_frames:
            self.epsilon = self.a/(self.a+self.frames_seen)
            epsilon = self.epsilon
        elif self.train:
            epsilon = 0.1

        if self.train:
            self.observation_history.append(observation)
            self.previous_state = self.new_state
            self.new_state = tensor_from_multiple_observations(self.observation_history[-4:])
        if random.random() < epsilon:
            return np.random.choice([0,1,2]) #Early exit
        else:
            with torch.no_grad():
                q_values = self.policy_net(self.new_state)
                return torch.argmax(q_values).item()


    def update_target_network(self):
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def store_transition(self, action, reward, done):
        action = torch.Tensor([[action]]).long().to(self.device)
        reward = torch.tensor([reward], dtype=torch.float32).to(self.device)
        self.memory.push(self.previous_state, action, self.new_state, reward, done)
