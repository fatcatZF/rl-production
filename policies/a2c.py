import torch.nn as nn 
import torch.nn.functional as F

class MLP_A2C(nn.Module):
    """
    Multi-layer perceptron advantage actor-critic
    """
    def __init__(self, n_observation, n_action):
        """
        args:
          n_observation: dimensions of observation spaces
          n_action: number of actions
        """
        super(MLP_A2C, self).__init__()
        self.l1 = nn.Linear(n_observation, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.l2 = nn.Linear(256,256)
        self.bn2 = nn.BatchNorm1d(256)
        self.actor = nn.Linear(256, n_action)
        self.critic = nn.Linear(256,1)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight.data)
                m.bias.data.fill_(0.1)
            elif isinstance(m, nn.BatchNorm1d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        """
        args:
          x: observation, shape: [batch_size, 3]
        """
        x = F.elu(self.bn1(self.l1(x)))
        x = F.elu(self.bn2(self.l2(x)))
        logits = self.actor(x)
        V = self.critic(x)
        return logits, V  
        
