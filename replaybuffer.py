import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import Any
from random import sample
    
class ReplayBuffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = [None] * buffer_size
        self.idx = 0

    def insert(self, sars):
        self.buffer[self.idx % self.buffer_size] = sars
        self.idx += 1

    def sample(self, num_samples):
        if num_samples < min(self.idx, self.buffer_size):
            if self.idx < self.buffer_size:
                return sample(self.buffer[: self.idx], num_samples)
            return sample(self.buffer, num_samples)




        
        
        
