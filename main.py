import torch
import numpy as np
from tqdm import tqdm
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
from replaybuffer import ReplayBuffer
from typing import Any
from random import  random
import wandb
from utils_ddqn import Inputresizing
from model_ddqn import ConvModel
from dataclasses import dataclass
import argparse


wandb.init(project="rl_atari")

parser=argparse.ArgumentParser(description='RLATARIGAME')

parser.add_argument('--lr',type=float,default=0.001)
parser.add_argument('--batch_size',type=int,default=32)
parser.add_argument('--rb_size',type=int,default=1000000)
parser.add_argument('--gamma',type=float,default=0.99)
parser.add_argument('--eps_decay',type=float,default=0.999999)
parser.add_argument('--seed',type=int,default=1)
parser.add_argument('--epoch',type=int,default=3000)
parser.add_argument('--eps_min',type=float,default=0.01)
parser.add_argument('--tgt_update',type=int,default=3000)

args=parser.parse_args()

wandb.config.update(args)

@dataclass
class Sarsd:
    state: Any
    action: int
    reward: float
    next_state: Any
    done: bool
    
    
def update_tgt_model(m, tgt):
    tgt.load_state_dict(m.state_dict())


def train_ddqn(model, state_transitions, tgt, num_actions, device, gamma):
    
    
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    
    actions = [s.action for s in state_transitions]
    model.opt.zero_grad()
    
    qvals = model(cur_states)  
    qvals=qvals.gather(1,torch.tensor(actions).unsqueeze(-1).to(device))
    
    greedy_qvals = model(next_states)
    greedy_actions=torch.argmax(greedy_qvals,dim=1)
    
    with torch.no_grad():
        qvals_next = tgt(next_states)  
        target_qvals=qvals_next.gather(1,greedy_actions.unsqueeze(-1))
    
    
    vals=rewards +((mask[:, 0] * target_qvals * gamma)[:,0].unsqueeze(1))
    

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(
        qvals, vals
    )

    loss.backward()
    model.opt.step()
    return loss

def train_dqn(model, state_transitions, tgt, num_actions, device, gamma):
    
    
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(device)
    
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(device)
    
    mask = torch.stack(([torch.Tensor([0]) if s.done else torch.Tensor([1]) for s in state_transitions])).to(device)
    
    next_states = torch.stack(([torch.Tensor(s.next_state) for s in state_transitions])).to(device)
    
    actions = [s.action for s in state_transitions]
    

    with torch.no_grad():
        qvals_next = tgt(next_states).max(-1)[0] 

    model.opt.zero_grad()
    qvals = model(cur_states) 
    one_hot_actions = F.one_hot(torch.LongTensor(actions), num_actions).to(device)

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(
        torch.sum(qvals * one_hot_actions, -1), rewards.squeeze() + mask[:, 0] * qvals_next * 0.99
    )

    loss.backward()
    model.opt.step()
    return loss

def run_test(model, env, device, max_steps=1000): 
    frames = []
    obs = env.reset()
    frames.append(env.frame)

    idx = 0
    done = False
    reward = 0
    while not done and idx < max_steps:
        action = model(torch.Tensor(obs).unsqueeze(0).to(device)).max(-1)[-1].item()
        obs, r, done, _ = env.step(action)
        reward += r
        frames.append(env.frame)
        idx += 1

    return reward, np.stack(frames, 0)


def main(chkpt=None, device="cuda",dqn=False,ddqn=False):
    """ 
    params
    
    rb_size = buffer size
    min_rb_size = min buffer size
    batch_size = sample size
    lr = learning rate
    eps_min = min epsilon
    eps_decay = epsilon decay rate
    env_steps_before_train = before train, the steps env needs to take
    tgt_model_update = interval to update target network
    epochs_before_test = before test, the steps env needs to take
    steps_since_train = after train, the steps env took
    step_num = the number of steps
    reward_history = the whold history of rewards
    episode_reward = rewards for one episode
    episodes=the number of episodes
    """
    rb_size = args.rb_size
    min_rb_size = 50000
    batch_size = args.batch_size
    lr = args.lr
    eps_min = args.eps_min
    eps_decay = args.eps_decay
    env_steps_before_train = 16
    tgt_model_update = args.tgt_update
    epochs_before_test = 1500
    steps_since_train = 0
    step_num = -1 * min_rb_size
    reward_history = []
    episode_reward = 0
    episodes=0
    

    env = gym.make("PongDeterministic-v4")
    env.seed(args.seed)
    env = Inputresizing(env, 84, 84, 4)

    test_env = gym.make("PongDeterministic-v4")
    test_env.seed(args.seed+1)
    test_env = Inputresizing(test_env, 84, 84, 4)
    
    

    last_observation = env.reset()

    m = ConvModel(env.observation_space.shape, env.action_space.n, lr).to(device)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = ConvModel(env.observation_space.shape, env.action_space.n,lr).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer(buffer_size=rb_size)
    

    tq = tqdm()
    try:
        while True:
            
            tq.update(1)

            eps = max(eps_decay ** (step_num),eps_min)
                
            if random() < eps:
                action = (
                    env.action_space.sample()
                )
            else:
                action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            episode_reward += reward

            rb.insert(Sarsd(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                reward_history.append(episode_reward)
                observation = env.reset()
                episodes+=1
                episode_reward = 0
                

            steps_since_train += 1
            step_num += 1

            if ( rb.idx > min_rb_size and steps_since_train > env_steps_before_train):
                if(ddqn):   
                    loss = train_ddqn(
                        m, rb.sample(batch_size), tgt, env.action_space.n, device,args.gamma
                    )
                elif(dqn):
                    loss = train_dqn(
                        m, rb.sample(batch_size), tgt, env.action_space.n, device,args.gamma
                    )
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "eps": eps,
                        "avg_reward" : np.mean(reward_history[-100:])
                    },
                    step=step_num,
                )
                


                if step_num % epochs_before_test==0:
                    rew, frames = run_test(m, test_env, device)
                    wandb.log({'test_reward': rew, 'test_video': wandb.Video(frames.transpose(0, 3, 1, 2), str(rew), fps=25, format='mp4')})
                    

                if step_num % tgt_model_update==0:
                    print("updating target model")
                    update_tgt_model(m, tgt)
                    torch.save(tgt.state_dict(), f"/home/labuser/doubledqn/ataritutorial/models/{step_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main(dqn=True)