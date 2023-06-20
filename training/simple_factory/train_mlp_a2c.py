import argparse
import os 
import time 

from collections import deque

import torch 
import torch.nn.functional as F 
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
    parser.add_argument("--resource-init", type=int, default=500)
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
            Qs.insert(0,discounted_rewards_sum+(gamma**est_depth)*value)

    return Qs 




            
def train(num_episodes=50000, gamma=0.99, est_depth=5, 
          lr=5e-4, actor_coeff=1, critic_coeff=0.5,
          entropy_coeff=0.01, max_grad_norm=0.5, num_episodes_update=5, resource_init=500):
    env = SimpleFactoryGymEnv(resource_init=resource_init)
    n_obs = 3
    n_action = env.action_space.n 
    mlp_a2c = MLP_A2C(n_obs, n_action) 

    optimizer = optim.Adam(mlp_a2c.parameters(), lr=lr)
    optimizer.zero_grad()

    for episode in range(num_episodes):
        rewards = []
        log_probs = []
        values = []
        entropies = []

        reward_episode = 0

        obs, info = env.reset() # reset the environment

        terminated = False

        while not terminated:
            state = torch.from_numpy(obs).float()
            mlp_a2c.eval()
            logits, v = mlp_a2c(state.unsqueeze(0)) #add batch dimension
            action = Categorical(logits=logits.squeeze()).sample().item()
            entropy = Categorical(logits=logits.squeeze()).entropy()
            entropies.append(entropy)
            log_prob = F.log_softmax(logits, dim=-1).squeeze()[action]
            log_probs.append(log_prob)
            values.append(v.squeeze())
            obs, reward, terminated, _, info = env.step(action)
            rewards.append(reward)
            reward_episode += reward 
        
        mlp_a2c.train()

        log_probs = torch.stack(log_probs)
        values = torch.stack(values)
        entropies = torch.stack(entropies) 
        values_list = values.tolist()
        Qs = torch.tensor(compute_Q_episode(rewards, values_list, gamma, est_depth)).float()
        As = Qs-values #compute advantages
        actor_loss = -(As*log_probs).mean()/num_episodes_update
        critic_loss = As.pow(2).mean()/num_episodes_update
        entropy_loss = entropies.mean()/num_episodes_update

        loss = actor_coeff*actor_loss + \
               critic_coeff*critic_loss + \
               entropy_coeff*entropy_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(mlp_a2c.parameters(),max_grad_norm)

        if (episode+1) % num_episodes_update==0:
            optimizer.step()
            optimizer.zero_grad()
        
            # Test
            #reward_episode = 0
            #with torch.no_grad():
            #    obs, info = env.reset() # reset the environment
            #    terminated = False
            #    while not terminated:
            #        state = torch.from_numpy(obs).float()
            #        mlp_a2c.eval()
            #        logits, v = mlp_a2c(state.unsqueeze(0)) #add batch dimension 
            #        action = torch.argmax(logits.squeeze()).item() #get deterministic action
            #        obs, reward, terminated, _, info = env.step(action)
            #        reward_episode += reward 

        print("episode: {}, reward: {}".format(episode+1, reward_episode))




if __name__ == "__main__":
    args = parse_args()