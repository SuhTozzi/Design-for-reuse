
import numpy as np
import math, random, json
from json import JSONEncoder
from collections import namedtuple, deque
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import gymnasium as gym

from dqn.dqn_model import DQN
from dqn.dqn_learn import Optimization


plt.ion()

# if GPU is to be used
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

class main():
    
    def __init__(
        self,
        env,
        EPISODE = 5, 
        BATCH_SIZE = 1,
        EPS_START = 0.9, 
        EPS_END = 0.05,
        EPS_DECAY = 10000,
        TAU = 0.0001,
        LR = 5e-4,
        DECAY_RATE = 0.99,
        MAX_STEPS = 5        
    ):

        Optimization(
            env = env, 
            q_func = DQN, 
            device = device, 
            EPISODE = EPISODE, 
            BATCH_SIZE = BATCH_SIZE,
            EPS_START = EPS_START, 
            EPS_END = EPS_END,
            EPS_DECAY = EPS_DECAY,
            TAU = TAU,
            LR = LR,
            DECAY_RATE = DECAY_RATE,
            MAX_STEPS = MAX_STEPS
        ) # DQN

