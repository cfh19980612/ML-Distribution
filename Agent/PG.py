import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Bernoulli
from torch.autograd import Variable
from itertools import count
import numpy as np
import pdb
from torch.distributions import Categorical



class PolicyNet(nn.Module):
    def __init__(self):
        super(PolicyNet, self).__init__()

        self.fc1 = nn.Linear(57, 114)
        self.fc1.weight.data.normal_(0, 0.1)   # initialization
        # self.fc2 = nn.Linear(114, 50)
        # self.fc2.weight.data.normal_(0, 0.1)   # initialization
        self.fc3 = nn.Linear(114, 6)  # Prob of Left
        self.fc3.weight.data.normal_(0, 0.1)   # initialization
    def forward(self, x):
        x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x),dim=0)
        return x

class PG(object):
    def __init__(self):
        self.batch_size = 10
        self.learning_rate = 0.05
        self.gamma = 0.99
        self.policy_net = PolicyNet()
        self.optimizer = torch.optim.RMSprop(self.policy_net.parameters(), lr=self.learning_rate)
        self.steps = 0
        self.episode_durations = []
        self.state_pool = []
        self.action_pool = []
        self.reward_pool = []
    def choose_actions(self, state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        probs = self.policy_net.forward(state)
        m = Categorical(probs)
        action = m.sample() 
        return action,probs
    
    def store_transition(self, state, action, reward, next_state):
        state = torch.from_numpy(state).float()
        state = Variable(state)
        self.state_pool.append(state)
        self.action_pool.append(float(action))
        self.reward_pool.append(reward)
        state = next_state
        state = torch.from_numpy(state).float()
        state = Variable(state)
        self.steps += 1

    def learn(self,e):
        if e > 30 and e % self.batch_size == 0:
            # Discount reward
            running_add = 0
            for i in reversed(range(self.steps)):
                if self.reward_pool[i] == 0:
                    running_add = 0
                else:
                    running_add = running_add * self.gamma + self.reward_pool[i]
                    self.reward_pool[i] = running_add

            # Normalize reward
            reward_mean = np.mean(self.reward_pool)
            reward_std = np.std(self.reward_pool)
            for i in range(self.steps):
                self.reward_pool[i] = (self.reward_pool[i] - reward_mean) / reward_std

            # Gradient Desent
            self.optimizer.zero_grad()

            for i in range(self.steps):
                state = self.state_pool[i]
                action = Variable(torch.FloatTensor([self.action_pool[i]]))
                reward = self.reward_pool[i]

                #probs = self.policy_net(state)
                probs = self.policy_net.forward(state)
                m = Categorical(probs)
                action = m.sample() 

                loss = -m.log_prob(action) * reward  # Negtive score function x reward
                loss.backward()

            self.optimizer.step()

            self.state_pool = []
            self.action_pool = []
            self.reward_pool = []
            self.steps = 0


