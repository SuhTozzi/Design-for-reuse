
import numpy as np
import math, random, json
from json import JSONEncoder
from collections import namedtuple, deque
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from dqn import utils
import time

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))
# plt.ion()


class Optimization():

    def __init__(
            self,
            env, q_func, device, 
            EPISODE = 5, BATCH_SIZE = 1,
            EPS_START = 0.9, EPS_END = 0.05, EPS_DECAY = 10000,
            TAU = 0.0001, LR = 5e-4, DECAY_RATE = 0.99,
            MAX_STEPS = 5 
    ):
        # Record the start time
        self.start_time = time.time()

        self.env = env
        self.q_func = q_func
        self.device = device
        if torch.cuda.is_available() or torch.backends.mps.is_available():
            self.num_episodes = EPISODE
        else:
            self.num_episodes = 5   # Check GPU
        self.BATCH_SIZE = BATCH_SIZE
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR
        self.DECAY_RATE = DECAY_RATE
        self.MAX_STEPS = MAX_STEPS

        ## BUILD DQN MODEL
        # self.n_observations = len(self.env.state)           # 1D
        self.n_observations = np.shape(self.env.state)      # 2D
        self.n_actions = self.env.action_space.n

        ## Construct the replay memory
        self.policy_net = q_func(self.n_observations, self.n_actions).to(self.device)
        self.target_net = q_func(self.n_observations, self.n_actions).to(self.device)

        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = utils.ReplayMemory(1000000)     # 100000  # 1000000   # 2000000

        # Record the process
        self.info_stack = {
            "HYPERPARAM": {
                "EPISODE":self.num_episodes, "BATCH_SIZE":self.BATCH_SIZE,
                "EPS_START":self.EPS_START, "EPS_END":self.EPS_END, "EPS_DECAY":self.EPS_DECAY,
                "TAU":self.TAU, "LR":self.LR, "DECAY_RATE":self.DECAY_RATE, 
                "MAX_STEPS":self.MAX_STEPS
        }
        }
        self.key_rewards = []

        self.dqn_learning()

    def dqn_learning(self):

        max_reward = -100
        ep_durations, ep_rewards = [], []

        for i_episode in range(self.num_episodes):

            self.steps_done = 0
            # Initialize the Env and get its state
            state = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            info_ep, rewards_ep = [], []
            for c in count():
                action = self.select_action(state)
                observation, reward, terminated, truncated, info = self.env.step(action.item())
                done = terminated or truncated

                reward = torch.tensor([reward], device=self.device)
                if terminated:
                    next_state = None
                else:
                    next_state = torch.tensor(observation, 
                                              dtype=torch.float32, device=self.device).unsqueeze(0)

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)
                rewards_ep.append(float(reward))

                # Move to the next state
                state = next_state

                # Perform one step of the optimization on the policy network
                self.optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU \
                        + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                info_ep.append(info)    # info_ep = [{info for each step}]

                if done:
                    final_reward = round(np.sum(rewards_ep)/self.steps_done, 4)

                    ep_durations.append(int(i_episode+1))
                    ep_rewards.append(final_reward)
                    info_ep.append({
                        "avg reward": final_reward
                    })

                    if ep_rewards[-1] > max_reward:
                        self.info_stack[f"episode_{i_episode}"] = info_ep
                        max_reward = float(ep_rewards[-1])
                        self.key_rewards.append(i_episode)

                    utils.plot_rewards(ep_dur=ep_durations, ep_rs=ep_rewards)

                    break

        end_time = time.time()
        elapsed_time = round(end_time - self.start_time, 2)
        self.info_stack[f"episode_{self.key_rewards[-1]}"][-1]["time"] = elapsed_time
        print(f"Reward: {max_reward}  |  Time: {elapsed_time} s")

        file_name = utils.export_results(self.info_stack, self.key_rewards)
        utils.plot_rewards(ep_dur=ep_durations, ep_rs=ep_rewards, show_result=True, file_name=file_name)
        
        plt.ioff()
        
    def select_action(self, state):
        sample = random.random()
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1
        if sample > eps_threshold**2:
            with torch.no_grad():
                return self.policy_net(state).max(1).indices.view(1,1)
        else:
            return torch.tensor([[self.env.action_space.sample()]], 
                                device=self.device, dtype=torch.long)

    def optimize_model(self):
        if len(self.memory) < self.BATCH_SIZE:
            return -1
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch, convering batch-array of Transitions to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None, batch.next_state)),
                                    device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                        if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then
        # we select the columns of actions taken which would've been taken for each batch state according to policy_net
        state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        with torch.no_grad():
            next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.DECAY_RATE) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss= criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

        return

