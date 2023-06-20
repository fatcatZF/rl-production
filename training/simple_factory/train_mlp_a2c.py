import argparse
import os 
import time 

import torch 
from torch.distributions import Categorical
import torch.optim as optim 


from environments.simple_factory import SimpleFactoryGymEnv


#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def parse_args():
    parser = argparse.ArgumentParser("MLP A2C for Simple Factory")
    parser.add_argument("--seed", type=int, default=42, help="random seed")
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--log-dir", type=str, default="./logs", help="Where to save log files")
    parser.add_argument("--num-steps", type=int, default=5, help="number of Monte Carlo steps for estimating TD-error")
    parser.add_argument("--num-episodes-update", type=int, default=5, help="number of episodes to update policy")
    parser.add_argument("--actor-loss-coeff", type=float, default=1.0, help="actor loss coefficient")
    parser.add_argument("--critic-loss-coeff", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--entropy-loss-coefficient", type=float, default=0.01, help="entropy loss coefficient (for exploration)")
    return parser.parse_args()







if __name__ == "__main__":
    args = parse_args()