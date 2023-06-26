import argparse
import os 
import time 
import datetime

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
    parser.add_argument("--log-dir", type=str, default="training/simple_factory/logs", help="Where to save log files")
    parser.add_argument("--est-depth", type=int, default=5, help="number of Monte Carlo steps (estimation depth) for estimating TD-error")
    parser.add_argument("--num-episodes-update", type=int, default=5, help="number of episodes to update policy")
    parser.add_argument("--actor-loss-coeff", type=float, default=1.0, help="actor loss coefficient")
    parser.add_argument("--critic-loss-coeff", type=float, default=0.5, help="critic loss coefficient")
    parser.add_argument("--entropy-loss-coeff", type=float, default=0.01, help="entropy loss coefficient (for exploration)")
    parser.add_argument("--lr", type=float, default=5e-4, help="learning rate for optimization")
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="maximum norm of gradients")
    parser.add_argument("--num-episodes", type=int, default=500, help="number of episodes for training")
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
          entropy_coeff=0.01, max_grad_norm=0.5, num_episodes_update=5,
          log_dir="training/simple_factory/logs"):
    env = SimpleFactoryGymEnv()
    n_obs = 3
    n_action = env.action_space.n 
    mlp_a2c = MLP_A2C(n_obs, n_action) 
  
    # Save Location
    now = datetime.datetime.now()
    timestamp = now.isoformat()
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)
    save_folder = "{}/{}".format(log_dir, timestamp)
    os.mkdir(save_folder)
    print("policy will be saved at {}".format(save_folder))
    policy_file = os.path.join(save_folder, "mlp_a2c.pt")

    optimizer = optim.Adam(mlp_a2c.parameters(), lr=lr)
    optimizer.zero_grad()

    rewards_episode = deque(maxlen=num_episodes_update)
    average_rewards_episode_max = 0.

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
        
        rewards_episode.append(reward_episode)

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

        print("episode: {}, reward: {:.2f}, finish time: {:.2f}, total products: {}".format(episode+1, reward_episode, info["current_time"], info["products"]))

        if (episode+1) % num_episodes_update==0:
            average_rewards_episode = np.mean(rewards_episode)
            if average_rewards_episode > average_rewards_episode_max:
                average_rewards_episode_max = average_rewards_episode
                print("Best Average Episode Reward: {:.2f}".format(average_rewards_episode_max))
                torch.save(mlp_a2c, policy_file)
                print("Best Policy So Far, Saving...")

            optimizer.step()
            optimizer.zero_grad()
        
            

        




if __name__ == "__main__":
    args = parse_args()
    print(args)
    train(args.num_episodes, args.gamma, args.est_depth, args.lr, 
          args.actor_loss_coeff, args.critic_loss_coeff, args.entropy_loss_coeff,
          args.max_grad_norm, args.num_episodes_update, args.log_dir)
