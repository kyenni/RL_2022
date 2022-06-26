# RL_2022
Deep reinforcement learning 2022

Trained and tested on:
```python

Python 3.8.3 
PyTorch 1.11.0
CUDA 9.1
gym 0.24.1 

```
# Model
## DQN

## Double DQN


# Experiment
* Optimizer : Adams
* Input : 84*84*4
210*160사이즈의 입력을 84*84로 size 변경 후, 4개의 frame을 입력으로 사용


## PONG
### Hyperparameters
lr=0.001
batch_size=32
rb_size=1000000
gamma=0.99
eps_decay=0.999999
seed=1
epoch=3000
eps_min=0.01
tgt_update=3000

