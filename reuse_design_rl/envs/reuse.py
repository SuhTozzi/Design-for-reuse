
import numpy as np
import math, random, copy

import gymnasium as gym
from gymnasium import spaces

from utils import reward as Reward


class ReuseEnv(gym.Env):

    def __init__(self, config={}):
        self.outer_arr = config.get("outer_arr")
        self.w = self._get_width
        self.h = self._get_height
        self.inventory_original = config.get("inventory", {0:np.array([0])})
        self.max_inv_id = len(self.inventory_original)
        self.max_steps = config.get("max_steps", 50)
        canvas_low = np.zeros((self.h, self.w))
        canvas_high = np.ones((self.h, self.w))*(100+self.max_inv_id)
        inv_low = np.ones([self.max_inv_id])*(-1)
        inv_high = np.ones([self.max_inv_id])*2     # inv_used
        
        self.canvas_area = self.w * self.h

        # Define Spaces
        self.observation_space = spaces.Dict(
            {
                "inventory": spaces.Box(    # This will be just a key list to call the actual inventory info
                    low = inv_low,
                    high = inv_high,
                    dtype = np.int32
                ),
                "canvas": spaces.Box(       # np.array()
                    low = canvas_low,
                    high = canvas_high,
                    dtype = np.int32
                )
            }
        )
        self.action_space = spaces.Discrete(8, seed=42)

        self.reset()

    def _get_obs(self, action=-1):
        ## General info + Canvas + Current stock
        # Canvas + Current stock
        cur_stock_shape = self.inventory[self.inv_id]
        ch, cw = np.shape(cur_stock_shape)
        stock_left_cut, stock_right_cut = [0,0], [0,0]  # [h,w]
        ref_yx_temp = copy.deepcopy(self.ref_yx)
        if np.any([ref_pt > [self.h,self.w][i] for i, ref_pt in enumerate(ref_yx_temp)]):
            cur_stock_shape_adj = np.array([])
            cur_stock_shape_padded = np.zeros((self.h,self.w))
            ref_pt = np.array([0,0])
        else:
            for i, ref_pt in enumerate(ref_yx_temp):
                if ref_pt < 0:
                    stock_left_cut[i] = abs(ref_pt)
                    ref_yx_temp[i] = 0
                if ref_pt + np.shape(cur_stock_shape)[i] > [self.h,self.w][i]:
                    stock_right_cut[i] = ref_pt + np.shape(cur_stock_shape)[i] - [self.h,self.w][i]
            cur_stock_shape_adj = cur_stock_shape[stock_left_cut[0]:ch-stock_right_cut[0],
                                                stock_left_cut[1]:cw-stock_right_cut[1]]
            cur_stock_shape_padded = np.pad(cur_stock_shape_adj, ((ref_yx_temp[0], self.h-np.shape(cur_stock_shape_adj)[0]-ref_yx_temp[0]), \
                                                                  (ref_yx_temp[1], self.w-np.shape(cur_stock_shape_adj)[1]-ref_yx_temp[1])))
        
        # General info
        obs_canvas = np.append(
            [self.step_count, action, np.count_nonzero(self.cur_canvas_state==0), Reward.concavity(canvas=self.cur_canvas_state, w=self.w)],
            [max(x)-min(x)+1 for x in list(np.where(self.cur_canvas_state==0))]
        )
        obs_inv = np.append(
            [self.inv_id, self.ref_yx[0], self.ref_yx[1], np.count_nonzero(self.inventory[self.inv_id]==1), Reward.concavity(canvas=self.inventory[self.inv_id], w=self.w)],
            np.append(np.shape(self.inventory[self.inv_id]), [len(self.avail_stocks), len(self.used_stocks)])
        )
        obs_combined = np.append(obs_canvas, obs_inv)

        pad_mul = [math.ceil(self.canvas_area/len(obs_combined))+1 if len(obs_combined) > self.canvas_area else 1][0]
        obs_combined_padded = np.pad(obs_combined, (0,self.canvas_area*pad_mul-len(obs_combined)))
        arr_padded_reshape = obs_combined_padded.reshape(pad_mul,(self.h,self.w)[0],-1)

        return np.concatenate((np.append([self.cur_canvas_state], [cur_stock_shape_padded], axis=0), arr_padded_reshape))

    def reset(self):
        self.step_count = 0
        self.cur_canvas_state = copy.deepcopy(self.outer_arr)
        self.inventory = copy.deepcopy(self.inventory_original)
        self.max_inv_id = len(self.inventory_original)

        ## Random
        self.avail_stocks = np.sort(
            np.random.choice(list(self.inventory.keys()), size=(self.max_inv_id*9)//10, replace=False)
        )
        ## Non-random
        # self.avail_stocks = np.array(list(self.inventory.keys()))

        self.inv_id = sorted([(np.count_nonzero(self.inventory[inv_key]==1), inv_key) for inv_key in self.avail_stocks], 
                             key = lambda x: x[0])[-1][-1]
        self.used_stocks = np.array([])
        self.ref_yx = np.array([0,0])
        self.state = self._get_obs()

        return self.state
    
    def step(self, action):
        self.step_count += 1
        terminated, truncated, reward = False, False, 0

        # Set the canvas
        cur_canvas_state = self.cur_canvas_state.copy()

        # Set the inv & current stock
        ref_yx = self.ref_yx.copy()
        cur_stock = self.inventory[self.inv_id]
        cur_stock_shape = cur_stock.copy()

        # 0: Change the stock
        if action == 0:
            if len(self.avail_stocks) > 1:
                if np.round(np.random.rand(1),1) < 0.8:
                    self.inv_id = int(sorted([
                        (np.count_nonzero(self.inventory[inv_key]==1), inv_key) for inv_key in self.avail_stocks[self.avail_stocks != self.inv_id]
                    ], key = lambda x: x[0])[-1][-1])
                else:
                    self.inv_id = int(np.random.choice(self.avail_stocks[self.avail_stocks != self.inv_id], 1))
                cur_stock_shape = self.inventory[self.inv_id]

                reward += Reward.velocity(cur_canvas_state, cur_stock_shape, ref_yx)

            else:
                reward -= 10

        # 1-2: Geometry
        elif action < 3:
            """
            Action-1: Horizontal cut from the bottom
            Action-2: Vertical cut from the right
            """
            if np.count_nonzero(cur_stock==1) == 0:
                reward -= 10
            else:
                # Prepare for the cut-off availability test
                cut_i = np.where(cur_stock==1)[action-1].max()
                if action == 1:
                    new_stock_shape = cur_stock[cut_i:,:]
                else:
                    new_stock_shape = cur_stock[:,cut_i:]

                # Evaluation
                if np.count_nonzero(new_stock_shape) != 0:  # Yes Cut-off
                    org_cur_stock_area = np.count_nonzero(cur_stock_shape)
                    # Update cur_stock_shape & inv_DB
                    if action == 1:
                        cur_stock_shape = cur_stock[:cut_i,:]
                    else:
                        cur_stock_shape = cur_stock[:,:cut_i]

                    # Update inventory
                    self.inventory[self.inv_id] = cur_stock_shape
                    self.inventory[self.max_inv_id] = new_stock_shape
                    self.avail_stocks = np.append(self.avail_stocks, self.max_inv_id)
                    self.max_inv_id += 1

                    reward -= (np.count_nonzero(new_stock_shape)/org_cur_stock_area + 1)

                else:  # No Cut-off
                    reward -= 10

        # 3-6: Translation
        elif action < 7:
            if ref_yx[action//5] + (-1)**action < [self.h, self.w][action//5]:
                ref_yx[action//5] += (-1)**action   # Temporarily change the ref pt
            else:
                ref_yx[action//5] = [self.h-1, self.w-1][action//5]

            ## Check if the stock is within the range
            stock_h, stock_w = cur_stock_shape.shape
            if True in (ref_yx<0) or (ref_yx[0]+stock_h > self.h) or (ref_yx[1]+stock_w > self.w):
                reward -= 10
            else:
                stock_temp = cur_stock_shape + cur_canvas_state[ref_yx[0]:ref_yx[0]+stock_h,
                                                                ref_yx[1]:ref_yx[1]+stock_w]
                if True in (stock_temp > 1):    # No overlap
                    reward -= 10
                else:
                    self.ref_yx = ref_yx    # Update the ref pt
                    reward += Reward.velocity_concav(cur_canvas_state, cur_stock_shape, ref_yx)
            
        # 7: Place the stock
        else:
            #1# Check if the stock is within the range
            cur_stock_h, cur_stock_w = cur_stock_shape.shape

            if (ref_yx[0]+cur_stock_h <= self.h) and (ref_yx[1]+cur_stock_w <= self.w):
                cur_stock_shape_temp = cur_stock_shape + cur_canvas_state[ref_yx[0]:ref_yx[0]+cur_stock_h, ref_yx[1]:ref_yx[1]+cur_stock_w]
                #2# If not overlapped, locate the stock on the canvas
                if not True in (cur_stock_shape_temp > 1):
                    cur_stock_shape_temp = copy.deepcopy(cur_stock_shape)
                    cur_stock_shape_temp[cur_stock_shape==1] = self.inv_id+100
                    cur_canvas_state[ref_yx[0]:ref_yx[0]+cur_stock_h, 
                                     ref_yx[1]:ref_yx[1]+cur_stock_w] = cur_stock_shape_temp
                    self.avail_stocks = np.delete(self.avail_stocks, np.where(self.avail_stocks == self.inv_id))
                    self.used_stocks = np.append(self.used_stocks, self.inv_id)

                    ## REWARD
                    reward_concav, vertex_par = 0, np.array([0, self.w-1])
                    for r in range(cur_canvas_state.shape[0]):
                        if np.any(cur_canvas_state[r,:] >= 100):
                            c = np.argwhere(cur_canvas_state[r,:] >= 100)
                            vertex_cur = np.array([c.min(), c.max()])
                            if len(np.unique(c)) == 1 or list(np.unique(c)) != list(range(c.min(), c.max()+1)):
                                reward_concav -= np.unique(c).size    # Partially empty in the middle
                            if c.min() != 0:
                                reward_concav -= 1  # Empty first column
                            reward_concav -= np.count_nonzero(vertex_par!=vertex_cur)*2     # Min corners
                            vertex_par = vertex_cur
                        elif not np.any(cur_canvas_state[r,:] >= 100) and reward_concav != 0:
                            reward_concav -= 2  # Row fully empty in the middle
                    reward_concav = reward_concav/(self.w*self.h)   # More reuse with less shape complexity
                    
                    reward += 20*(10+np.count_nonzero(cur_stock_shape)/self.canvas_area + reward_concav)
                    
                    self.cur_canvas_state = cur_canvas_state

                    # Update for the next step
                    ## ref pt
                    if ref_yx[0] + cur_stock_h < self.h-1:
                        ref_yx[0] += cur_stock_h
                    elif ref_yx[1] + cur_stock_w < self.w:
                        ref_yx[1] += cur_stock_w
                        ref_yx[0] = 0
                    self.ref_yx = ref_yx
                    ## inv_id
                    if len(self.avail_stocks) > 0:
                        if np.round(np.random.rand(1),1) < 0.8:
                            self.inv_id = int(sorted(
                                [(np.count_nonzero(self.inventory[inv_key]==1), inv_key) for inv_key in self.avail_stocks[self.avail_stocks != self.inv_id]], 
                                key = lambda x: x[0]
                            )[-1][-1])
                        else:
                            self.inv_id = int(np.random.choice(self.avail_stocks[self.avail_stocks != self.inv_id], 1))
                else:
                    reward -= 10    # Stock overlapped
            else:
                reward -= 10    # Stock out-of-range
        
        # Terminate/Truncate
        if self.step_count >= self.max_steps:
            truncated = True
        if np.count_nonzero(cur_canvas_state==0) < self.w*self.h*0.1:
            reward += 500
            terminated = True
        elif len(self.avail_stocks) == 0:
            reward += 500
            terminated = True

        self.state = self._get_obs(action=action)
        info = {
            "step": self.step_count,
            "action": action,
            "reward": reward,
            "canvas state": cur_canvas_state,
            "inventory id": self.inv_id,
            "reference point XY": list(np.flip(self.ref_yx)),
            "inventory state": self.inventory
        }

        return self.state, reward, terminated, truncated, info

    @property
    def _get_width(self):
        return int(self.outer_arr.shape[1])

    @property
    def _get_height(self):
        return int(self.outer_arr.shape[0])
    
