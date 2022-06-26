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
![pont_dqn](https://user-images.githubusercontent.com/88640075/175803030-0b2f25be-ad0c-4582-b535-7300e909c740.png)  

![pong_dqn](https://user-images.githubusercontent.com/88640075/175803027-97039c70-ff1d-4e9d-bf47-6609bdd5bd47.png)  

![pong_dqn_loss](https://user-images.githubusercontent.com/88640075/175803028-3fb74513-cf24-4b6d-ab99-2446c6826413.png)  

 

