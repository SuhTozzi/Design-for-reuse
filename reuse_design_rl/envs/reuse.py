
import numpy as np
import math, random
from copy import deepcopy

import shapely
from shapely import Polygon
from shapely import MultiPolygon
from shapely.geometry import Polygon, MultiPoint, LineString
from shapely.ops import split

import matplotlib
import matplotlib.pyplot as plt
import io
from PIL import Image

import torch
from torchvision import transforms

import gymnasium as gym
from gymnasium import spaces

matplotlib.use("Agg")

class ReuseTetris(gym.Env):

    def __init__(self, config={}):
        
        # Canvas
        canvas_dim = config.get("canvas")
        self.canvas_w = canvas_dim[0]
        self.canvas_h = canvas_dim[1]
        self.pol_canvas = Polygon([(0,0), (self.canvas_w,0), (self.canvas_w,self.canvas_h), (0,self.canvas_h), (0,0)])
        self.canvas_area = self.pol_canvas.area

        # Stock inventory
        inv_config = config.get("inventory")
        self.inv_stocks = {}
        for id, stock_pts in inv_config.items():
            self.inv_stocks[id] = Polygon(stock_pts)
        self.id_max = int(id)
        self.inv_count_max = config.get("max_number_of_stocks")

        # Define Spaces
        self.action_space = spaces.Discrete(9)
        self.obs_space = spaces.Box(0, 255, (96, 96, 3), dtype = np.uint8)

        # User preference
        self.weights = np.array(config.get("weight"))
        self.rl_step_max = config.get("max_step")
        
        self.reset()
    
    def reset(self):

        # Canvas
        self.pol_canvas_diff = deepcopy(self.pol_canvas)    # Remaining canvas after placing stocks
        self.pol_union = Polygon()

        # Inventory
        self.pol_inv = deepcopy(self.inv_stocks)
        self.queue = StockQueue(id_max=self.id_max, count_max=self.inv_count_max)
        self.id_queue = self.queue.reset()
        self.id_assigned = [] # Inventory placed on canvas w/ most recent status

        # Start!
        self.step_count = 0
        self.reward = 0
        self.len_plaster = 0

        state = np.zeros((96,96,3), dtype=np.float64)
        transform = transforms.ToTensor()
        self.state = transform(state)
        self.info = {"stock_queue":self.id_queue}

        return self.state

    def step(self, action):

        self.step_count += 1
        self.id_active = self.queue.retrieve_active()
        pol_stock_cur = self.pol_inv[self.id_active]
        rewards = np.array([0, 0, 0], dtype=np.float64)   # [reuse, work-efficiency, computing]

        """ Actions
        # 0: Placement
        # 1: Next stock on queue
        # 2: Cut parallel to X-axis
        # 3: Cut parallel to Y-axis
        # 4: Rotate 90 CW
        # 5: Move Right
        # 6: Move Left
        # 7: Move Up
        # 8: Move Down
        """

        if action == 0:     # Placement
            
            # First check validity
            if not self.collision(pol_stock_cur):
                # Rewards are first computed
                rewards = self._get_rewards(pol_stock_cur)
                
                # Update Environments
                self.pol_canvas_diff = shapely.difference(self.pol_canvas_diff, pol_stock_cur)  # Canvas
                self.id_assigned.append(self.id_active)     # inventory
                self.queue.update_queue()

            # Penalty if invalid
            else:
                rewards[2] -= 0.1

        elif action == 1:   # Next stock on queue

            self.id_active = self.queue.get_next_stock()
            pol_stock_cur = self.pol_inv[self.id_active]
            rewards[2] -= 0.1

        elif action == 2 or action == 3:   # Cut

            # Reward prep
            eval_prev = abs(self.pol_canvas_diff.area - pol_stock_cur.area)
            
            # Operation
            if action == 2:     # Cut parallel to X-axis
                temp = list(set([y for _, y in list(pol_stock_cur.exterior.coords)]))
                if len(temp) == 2 and temp[1]-temp[0] > 1:
                    y2_stock_cur = temp[1]-1
                else:
                    y2_stock_cur = temp[1]
                cutting_line = LineString([(-100,y2_stock_cur), (100,y2_stock_cur)])

            else:               # Cut parallel to Y-axis
                temp = list(set([x for x, _ in list(pol_stock_cur.exterior.coords)]))
                if len(temp) == 2 and temp[1]-temp[0] > 1:
                    x2_stock_cur = temp[1]-1
                else:
                    x2_stock_cur = temp[1]
                cutting_line = LineString([(x2_stock_cur,-100), (x2_stock_cur,100)])

            pol_stocks_added = sorted(split(pol_stock_cur, cutting_line).geoms, key=lambda x: x.area)
            pol_stock_cur = pol_stocks_added.pop()

            # Reward
            eval_cur = abs(self.pol_canvas_diff.area - pol_stock_cur.area)
            if  eval_cur >= eval_prev:
                rewards[2] -= 1
            else:
                rewards[2] -= 0.5
                
            # Update Env
            self.pol_inv[self.id_active] = pol_stock_cur

            for pol_temp in pol_stocks_added:
                id = self.queue.add_queue()
                pol_temp = shapely.affinity.translate(pol_temp,-pol_temp.bounds[0],-pol_temp.bounds[1]) # Move back to (0,0)
                self.pol_inv[id] = pol_temp

        elif action == 4:   # Rotation

            # Reward prep
            eval_prev = abs(self.pol_union.bounds[2]-pol_stock_cur.bounds[0] + self.pol_union.bounds[3]-pol_stock_cur.bounds[1])
            if not self.collision(pol_stock_cur):
                rewards[2] -= 1
            
            # Operation
            temp = shapely.affinity.rotate(pol_stock_cur, 90)
            pol_stock_cur = shapely.affinity.translate(
                temp,
                -temp.bounds[0],
                -temp.bounds[1]
            )

            # Reward
            eval_cur = abs(self.pol_union.bounds[2]-pol_stock_cur.bounds[0] + self.pol_union.bounds[3]-pol_stock_cur.bounds[1])
            if self.collision(pol_stock_cur):
                if eval_cur >= eval_prev:
                    rewards[2] -= 1
            else:
                if eval_cur >= eval_prev:
                    rewards[2] -= 0.5
                else:
                    rewards[2] += 0.2

            # Update Env
            self.pol_inv[self.id_active] = pol_stock_cur

        elif action < 7:    # Translate X

            # Reward prep
            eval_prev = abs(self.pol_canvas_diff.bounds[2] - pol_stock_cur.bounds[2])
            if not self.collision(pol_stock_cur):
                rewards[2] -= 1

            # Operation
            pol_stock_cur = shapely.affinity.translate(pol_stock_cur,(-1)**(5-action),0)

            # Reward
            eval_cur = abs(self.pol_canvas_diff.bounds[2] - pol_stock_cur.bounds[2])
            if self.collision(pol_stock_cur):
                if eval_cur >= eval_prev:
                    rewards[2] -= 1
                else:
                    rewards[2] += 0.1
            else:
                rewards[2] -= 0.5

            # Update Env
            self.pol_inv[self.id_active] = pol_stock_cur

        else:               # Translate Y

            # Reward prep
            eval_prev = abs(self.pol_canvas_diff.bounds[3] - pol_stock_cur.bounds[3])
            if not self.collision(pol_stock_cur):
                rewards[2] -= 1

            # Operation
            pol_stock_cur = shapely.affinity.translate(pol_stock_cur,0,(-1)**(7-action))
            
            # Reward
            eval_cur = abs(self.pol_canvas_diff.bounds[3] - pol_stock_cur.bounds[3])
            if self.collision(pol_stock_cur):
                if eval_cur >= eval_prev:
                    rewards[2] -= 1
                else:
                    rewards[2] += 0.1
            else:
                rewards[2] -= 0.5
            
            # Update Env
            self.pol_inv[self.id_active] = pol_stock_cur

        self.state, rewards, terminate = self._get_obs(rewards, pol_stock_cur)
        self.reward = round(float(np.sum(rewards*self.weights)), 2)   # weighted sum

        self.info[f"step{self.step_count}"]={
            "action": action,
            "stock_id": self.id_active,
            "rewards":  [round(num, 2) for num in rewards]
        }
        
        return self.state, self.reward, terminate, self.info

    def _get_obs(self, rewards=np.array([0,0,0]), pol_stock_cur=False):

        state = self.viz_polygon(pol_stock_cur)     # 2D image

        if self.collision(pol_stock_cur):
            rewards[2] -= 0.1

        # termination
        if len(self.queue.queue) == 0 or self.pol_canvas_diff.area == 0:    # early termination
            terminate = True
            rewards[2] += 10
            self.info["stocks_used"] = self.id_assigned
            self.layout_image = self.viz_final_layout()

        elif self.step_count == self.rl_step_max:
            terminate = True
            self.info["stocks_used"] = self.id_assigned
            self.layout_image = self.viz_final_layout()

        else:
            terminate = False

        return state, rewards, terminate
    
    def _get_rewards(self, pol_stock_cur):
        
        pol_union = shapely.unary_union([self.pol_union, pol_stock_cur])   # pol_union.area = state

        if not pol_union.is_empty:  # Skip when Step == 0 (.reset)

            # reuse: (+) reused area
            r_reuse = round(30*pol_union.area/self.canvas_area,2)

            # work efficiency: (-) longer plastering (+) meeting canvas boundary
            r_work = self.pol_canvas.boundary.intersection(pol_union.boundary).length \
                - self.len_plaster - self.pol_inv[self.id_active].length

            if pol_union.geom_type == 'MultiPolygon':    # Non-overlapping polygons exist
     
                for polygon in pol_union.geoms:
                    for pol in polygon.interiors:     # (-) intricate points
                        r_work -= 2*len(pol.coords)
                    r_work -= 2*len(polygon.exterior.coords)
            
            else:   # Single unified polygon
                for pol in pol_union.interiors:     # (-) intricate points
                    r_work -= 2*len(pol.coords)
                r_work -= 2*len(pol_union.exterior.coords)

            r_work = round(10 * r_work / (16*self.canvas_area), 2) # normalization

        else:   # Step 0 for reset
            return np.array([0, 0, 0], dtype=np.float64)

        self.pol_union = pol_union
        self.len_plaster += self.pol_inv[self.id_active].length

        return np.array([r_reuse, r_work, 0], dtype=np.float64)
    
    def collision(self, pol_stock_cur):
        if shapely.covers(self.pol_canvas_diff, pol_stock_cur):
            return False
        else:
            return True
        
    def viz_polygon(self, pol_stock_cur=False):

        fig = plt.figure(1, figsize=(1,1), dpi=96)
        ax = fig.add_subplot(111, xlim=(0,self.canvas_w), ylim=(0,self.canvas_h), aspect="equal")

        if len(self.id_assigned) > 0:

            if self.pol_union.geom_type == 'MultiPolygon':    # Non-overlapping polygons exist
                for polygon in self.pol_union.geoms:
                    ax.fill(*polygon.exterior.xy, color="blue", alpha=0.4, zorder=1)
                    if polygon.interiors:
                        for interior in polygon.interiors:
                            ax.plot(*interior.xy, color='white', zorder=1)
            else:
                ax.fill(*self.pol_union.exterior.xy, color="blue", alpha=0.4, zorder=1)
                if self.pol_union.interiors:
                    for interior in self.pol_union.interiors:
                        ax.plot(*interior.xy, color='white', zorder=1)
        else:
            ax.plot()

        if pol_stock_cur:
            ax.fill(*pol_stock_cur.exterior.xy, color="red", alpha=0.3, zorder=1)
            if pol_stock_cur.interiors:
                for interior in pol_stock_cur.interiors:
                    ax.fill(*interior.xy, color='white', alpha=0.3, zorder=1)

        ax.set_xticks([])
        ax.set_yticks([])

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert("RGB")
        image = image.resize((96, 96))
        image_array = transforms.ToTensor()(image)
        buf.close()
        plt.close(fig)

        return image_array

    def viz_final_layout(self):
        
        fig = plt.figure(1, figsize=(5,5), dpi=96)
        ax = fig.add_subplot(111, xlim=(0,self.canvas_w), ylim=(0,self.canvas_h), aspect="equal")

        if len(self.id_assigned) > 0:
            # Address all stocks assigned
            for id in self.id_assigned:
                # Draw
                polygon = self.pol_inv[id]
                ax.fill(*polygon.exterior.xy, color="blue", alpha=0.4, zorder=1)
                if polygon.interiors:
                    for interior in polygon.interiors:
                        ax.plot(*interior.xy, color='white', zorder=1)
                # Annotate
                x, y = tuple(polygon.representative_point().coords)[0]
                ax.text(x, y, f'{id}')
        else:
            ax.plot()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        image = Image.open(buf).convert('RGB')
        
        return image

class StockQueue:

    def __init__(self, id_max=5, count_max=5):
        self.id_max1 = id_max
        self.count_max = count_max
        
    def reset(self):
        # random.seed(42)         # If fixed availability, comment this out; vice versa
        queue = list(range(1,self.id_max1+1))
        # random.shuffle(queue)   # If fixed availability, comment this out; vice versa
        self.queue = queue[:self.count_max]
        self.id_max = deepcopy(self.id_max1)
        return self.queue

    def get_queue(self):
        return self.queue
    
    def update_queue(self):
        self.queue.pop(0)

    def add_queue(self):
        self.id_max += 1
        self.queue.append(self.id_max)
        return str(self.id_max)

    def get_next_stock(self):
        temp = self.queue[0]
        self.queue.pop(0)
        self.queue.append(temp)
        return str(self.queue[0])
    
    def retrieve_active(self):
        return str(self.queue[0])

