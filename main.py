import torch
import numpy as np
from tqdm import tqdm
import gym
import time
import torch.nn as nn
import torch.nn.functional as F
from agent_ddqn import ReplayBuffer
from typing import Any
from random import  random
import wandb
from utils_ddqn import FrameStackingAndResizingEnv
from models_ddqn import ConvModel
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


def train(model, state_transitions, tgt, num_actions, device, gamma=0.99):
    
    
    cur_states = torch.stack(([torch.Tensor(s.state) for s in state_transitions])).to(
        device
    )
    
    rewards = torch.stack(([torch.Tensor([s.reward]) for s in state_transitions])).to(
        device
    )
    mask = torch.stack(
        (
            [
                torch.Tensor([0]) if s.done else torch.Tensor([1])
                for s in state_transitions
            ]
        )
    ).to(device)
    next_states = torch.stack(
        ([torch.Tensor(s.next_state) for s in state_transitions])
    ).to(device)
    actions = [s.action for s in state_transitions]
    
    model.opt.zero_grad()
    qvals = model(cur_states)  # (N, num_actions)
    qvals=qvals.gather(1,torch.tensor(actions).unsqueeze(-1).to(device))
    
    greedy_qvals = model(next_states)
    greedy_actions=torch.argmax(greedy_qvals,dim=1)
    
    with torch.no_grad():
        qvals_next = tgt(next_states)  # (N, num_actions)
        target_qvals=qvals_next.gather(1,greedy_actions.unsqueeze(-1))
    
    
    vals=rewards +((mask[:, 0] * target_qvals * gamma)[:,0].unsqueeze(1))
    

    loss_fn = nn.SmoothL1Loss()
    loss = loss_fn(
        qvals, vals
    )

    loss.backward()
    model.opt.step()
    return loss


def run_test(model, env, device, max_steps=1000):  # -> reward, movie?
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


def main(test=False, chkpt=None, device="cuda"):
    
    memory_size = args.rb_size
    min_rb_size = 50000
    sample_size = args.batch_size
    lr = args.lr

    # eps_max = 1.0
    eps_min = args.eps_min

    eps_decay = args.eps_decay

    env_steps_before_train = 16
    tgt_model_update = args.tgt_update
    epochs_before_test = 1500

    env = gym.make("Breakout-v0")
    env.seed(args.seed)
    env = FrameStackingAndResizingEnv(env, 84, 84, 4)

    test_env = gym.make("Breakout-v0")
    test_env.seed(args.seed+1)
    test_env = FrameStackingAndResizingEnv(test_env, 84, 84, 4)
    
    

    last_observation = env.reset()

    m = ConvModel(env.observation_space.shape, env.action_space.n, lr=lr).to(device)
    if chkpt is not None:
        m.load_state_dict(torch.load(chkpt))
    tgt = ConvModel(env.observation_space.shape, env.action_space.n).to(device)
    update_tgt_model(m, tgt)

    rb = ReplayBuffer(buffer_size=memory_size)
    steps_since_train = 0
    epochs_since_tgt = 0
    epochs_since_test = 0

    step_num = -1 * min_rb_size

    episode_rewards = []
    rolling_reward = 0
    episodes=0

    tq = tqdm()
    try:
        while True:
            if test:
                env.render()
                time.sleep(0.05)
            tq.update(1)

            eps = eps_decay ** (step_num)
            if test:
                eps = 0
                
            if random() < eps:
                action = (
                    env.action_space.sample()
                )  # your agent here (this takes random actions)
            else:
                action = m(torch.Tensor(last_observation).unsqueeze(0).to(device)).max(-1)[-1].item()

            observation, reward, done, info = env.step(action)
            rolling_reward += reward

            rb.insert(Sarsd(last_observation, action, reward, observation, done))

            last_observation = observation

            if done:
                episode_rewards.append(rolling_reward)
                if test:
                    print(rolling_reward)
                observation = env.reset()
                episodes+=1
                wandb.log(
                    {
                        "Ep_reward": rolling_reward,
                    },
                    step=episodes
                )
                rolling_reward = 0
                

            steps_since_train += 1
            step_num += 1

            if (
                (not test)
                and rb.idx > min_rb_size
                and steps_since_train > env_steps_before_train
            ):
                loss = train(
                    m, rb.sample(sample_size), tgt, env.action_space.n, device
                )
                wandb.log(
                    {
                        "loss": loss.detach().cpu().item(),
                        "eps": eps,
                        "avg_reward" : np.mean(episode_rewards[-100:])
                    },
                    step=step_num,
                )
                epochs_since_tgt += 1
                epochs_since_test += 1

                if epochs_since_test > epochs_before_test:
                    rew, frames = run_test(m, test_env, device)
                    # T, H, W, C
                    wandb.log({'test_reward': rew, 'test_video': wandb.Video(frames.transpose(0, 3, 1, 2), str(rew), fps=25, format='mp4')})
                    epochs_since_test = 0

                if epochs_since_tgt > tgt_model_update:
                    print("updating target model")
                    update_tgt_model(m, tgt)
                    epochs_since_tgt = 0
                    torch.save(tgt.state_dict(), f"/home/labuser/doubledqn/ataritutorial/models/{step_num}.pth")

                steps_since_train = 0

    except KeyboardInterrupt:
        pass

    env.close()


if __name__ == "__main__":
    main()