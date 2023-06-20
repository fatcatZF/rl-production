import argparse
import os 
import time 

from collections import deque

import torch 
from torch.distributions import Categorical
import torch.optim as optim 

import numpy as np 

from environments.simple_factory import SimpleFactoryGymEnv
from policies.a2c import MLP_A2C


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser("MLP A2C for Simple Factory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Where to save log files")
    parser.add_argument("--est-depth", type=int, default=5, help="number of Monte Carlo steps (estimation depth) for estimating TD-error")
    parser.add_argument("--num-episodes-update", type=int, default=5, help="number of episodes to update policy")
    parser.add_argument("--actor-loss-coeff", type=float, default=1.0, help="actor loss coefficient")
    parser.add_argument("--critic-loss-coeff", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--entropy-loss-coefficient", type=float, default=0.01, help="entropy loss coefficient (for exploration)")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate for optimization")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradients")
    parser.add_argument("--num-episodes", type=int, default=50000, help="number of episodes for training")
    return parser.parse_args()


def compute_Q_episode(rewards, values, gamma=0.99, est_depth=5):
    """Compute Q values of one episode with Bootstraping with estimation depth
       args:
         rewards: rewards of one episode
         values: values computed by critic of one episode
    """
    episode_length = len(rewards)
    assert episode_length >= est_depth

    est_queue = deque(maxlen=est_depth)
    est_queue.extend(rewards[-est_depth:])
    gammas = np.array([gamma**i for i in range(est_depth)])
    Qs = [np.sum(gammas[:est_depth-i]*np.array(est_queue)[-(est_depth-i):]) for i in range(est_depth)]
    # estimated Q values for the last steps within the estimation depth

    if episode_length > est_depth:
        for t in reversed(range(episode_length-est_depth)):
            est_queue.appendleft(rewards[t])
            value = values[t+est_depth]
            discounted_rewards_sum = np.sum(np.array(est_queue)*gammas)
            Qs.append(discounted_rewards_sum+(gamma**est_depth)*value)

    return Qs 
            




if __name__ == "__main__":
    args = parse_args()