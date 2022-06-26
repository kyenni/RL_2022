# RL_2022
__Deep reinforcement learning 2022__  

Trained and tested on:
```python

Python 3.8.3 
PyTorch 1.11.0
CUDA 9.1
gym 0.24.1 
wandb

```
# Model

## Q-network
* Deepmind의 deep reinforcement learning tutorial에서의 모델 구조를 참고   
 >
<img width="60%" src="https://user-images.githubusercontent.com/88640075/175806446-05ae242a-06bd-49ee-acb7-7434749d1985.png">   

## DQN    
<img width="60%" src="https://user-images.githubusercontent.com/88640075/175809338-847a1bbb-80b1-408f-bbee-0b21eafbee48.png">  

## Double DQN    

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175809339-38998efd-6ac1-4334-8fcd-b6ce505c9ac2.png">


# Experiment
* Optimizer : Adam
* Input : 84x84x4  
210x160x3 사이즈의 입력을 84x84로 size 변경 후, 4개의 frame을 입력으로 사용
* epsilon decay    
<img width="60%" src="https://user-images.githubusercontent.com/88640075/175806022-faa4b338-950c-40d9-8088-cb70c86a652b.png">  

## PongDeterministic-v4
__Hyperparameters__  
lr=0.0001  
batch_size=32  
rb_size=1000000  
gamma=0.99  
eps_decay=0.999999  
seed=1  
eps_min=0.01  
tgt_update=5000 

## DQN  

__Best score=10__
<p align="center">
<img src="https://user-images.githubusercontent.com/88640075/175805290-b38a4077-86d1-4f6a-bdc2-56df9bb360a9.gif">
</p>

__train reward__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175803030-0b2f25be-ad0c-4582-b535-7300e909c740.png"/>  

__test reward__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175803027-97039c70-ff1d-4e9d-bf47-6609bdd5bd47.png"/>
 
__loss__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175803028-3fb74513-cf24-4b6d-ab99-2446c6826413.png"/>

## Double DQN
__Best score=1__

<p align="center">
<img src="https://user-images.githubusercontent.com/88640075/175805520-73b63e17-1129-41ff-a787-05d55307daaa.gif">
</p>

__train reward__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175805510-d7ed2e9c-754a-4ac6-9e94-ffe90dfe9e79.png"/>  

__test reward__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175805512-16d3ae90-b198-46b5-9bec-271c2327adb4.png"/>
 
__loss__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175805508-20d22fd1-8f4b-4b6a-b16a-b6adf8ecdeb7.png"/>

## SpaceInvaders-v0
__Hyperparameters__  

lr=0.0001  
batch_size=32  
rb_size=1000000  
gamma=0.99  
eps_decay=0.999999  
seed=1  
eps_min=0.01  
tgt_update=5000 
 
## DQN  

__Best score=555__

<p align="center">
<img src="https://user-images.githubusercontent.com/88640075/175805679-30bb2ab5-0b9e-4249-a4d8-3406e29dba37.gif">
</p>

__train reward__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175805675-b6eadace-18b1-47bb-b38c-537cf542c710.png"/>  

__test reward__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175805678-9814a004-3264-4d61-9bc8-f57aa15d68fe.png"/>
 
__loss__  

<img width="60%" src="https://user-images.githubusercontent.com/88640075/175805672-7f992cc3-5d20-488b-8399-764392cd2866.png"/>

# 한계점
* score가 점진적으로 증가하지 않고 지속적으로 변동이 심하여 수렴하는 형태를 보이지 않음
* dqn에 비해 double dqn이 불안정한 학습을 보여서 지속할 수가 없었음
* gamma를 0.99에서 낮추거나 lr를 변경하여 시도해보았지만 더욱더 변동성이 커지는 모습을 보였음
* 코드를 이해하느라 다른 게임에 대한 시도를 해보지 못하였음
* 이론적으로는 이해하였음에도 코드로 공부가 더 필요하다고 느꼈음
