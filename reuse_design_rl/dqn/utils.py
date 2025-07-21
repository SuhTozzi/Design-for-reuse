
import numpy as np
import random, json
from json import JSONEncoder
from datetime import datetime

from collections import namedtuple, deque
import torch

import matplotlib
import matplotlib.pyplot as plt

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display
plt.rcParams["figure.figsize"] = (5, 5)
plt.ion()


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    

class NumpyArrayEncoder(JSONEncoder):

    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return int(obj)
        return JSONEncoder.default(self, obj)
    

def plot_rewards(ep_dur=[], ep_rs=[], show_result=False, file_name="default_name"):
    fig = plt.figure(1,figsize=(10, 5))
    durations_t = torch.tensor(ep_dur, dtype=torch.int64)
    rewards_t = torch.tensor(ep_rs, dtype=torch.float)
    if show_result:
        plt.title("Result")
    else:
        plt.clf()
        plt.title("Training...")
    plt.plot(durations_t, rewards_t, c='royalblue')
    plt.xlabel("Episode")
    plt.ylabel("Avg Reward")
    plt.pause(0.001)
    if is_ipython:
        if not show_result:
            display.display(plt.gcf())
            display.clear_output(wait=True)
        else:
            # display.display(plt.gcf())
            fig.savefig(f'./result/{file_name}.jpg')
    return


def export_results(info_stack, key_rewards=None):

    encoded_info = json.dumps(info_stack, cls=NumpyArrayEncoder)
    file_name = f'result_{datetime.now().strftime("%Y%m%d-%H%M")}'

    with open(f'./result/{file_name}.json', 'w', encoding="utf-8") as f:
        json.dump(encoded_info, f)

    if key_rewards != None:
        print("Major Updates:", key_rewards)
        print(file_name)

    return file_name


