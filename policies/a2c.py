import torch.nn as nn 
import torch.nn.functional as F

class FC_A2C(nn.Module):
    def __init__(self, n_observation, n_action):
        """
        args:
          n_observation: dimensions of observation spaces
          n_action: number of actions
        """
        super(FC_A2C, self).__init__()
        self.l1 = nn.Linear(n_observation, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.l2 = nn.Linear(256,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.actor = nn.Linear(256, n_action)
        self.critic = nn.Linear(256,1)

    def forward(self, x):
        """
        args:
          x: observation, shape: [batch_size, 3]
        """
        x = F.elu(self.bn1(self.l1(x)))
        x = F.elu(self.bn2(self.l2(x)))
        actions = self.actor(x)
        value = self.critic(x)
        return actions, value 
        
