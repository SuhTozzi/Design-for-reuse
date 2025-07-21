import numpy as np
import random, json, glob, os
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.ticker import MultipleLocator


class PlotAnalysis():

    def __init__(self, file_name=None):

        if file_name == None:
            list_of_files = glob.glob(r'..\result\*.json')
            file_name = max(list_of_files, key=os.path.getmtime)
        else:
            file_name=r'..\result\{}.json'.format(file_name)

        self.file_name = file_name
        plt.ion()

        with open(self.file_name, 'r', encoding='utf-8') as f:
            data = json.load(f)
        self.data = json.loads(data)
        self.default_ep = int(list(self.data.keys())[-1].split('_')[1])

        self.max_steps = self.data["HYPERPARAM"]["MAX_STEPS"]
        self.c_chart = ['#163B8F','#168AAD','#76C893','#52B69A','#1E6091','#F0F3BD','#37999E','#D9ED92']

    def select_eps(self, count=None, must_have=[], last_ep=False):
        if count == None:
            count = min(6,len(self.data.keys())-1)
        all_eps = [int(ep.split('_')[-1]) for ep in list(self.data.keys())[1:]]
        result = []
        if last_ep:
            result.extend(all_eps[-1])
        elif count < 3:
            result.extend(random.sample(all_eps,count))
        else:
            result = [all_eps[0], all_eps[-1]]
            result[1:1]=sorted(random.sample(all_eps[1:-1],count-2))

        if must_have != []:
            result.extend(must_have)
            result = list(set(result))

        return sorted(result)     

    def show_layout(self, ep=None, step=-2, **kwargs):
        if ep == None:
            ep = self.default_ep
        print(ep)
        data_ep = np.array(self.data[f"episode_{ep}"][step]['canvas state'])
        
        # print(data_ep)
        cmap = colors.ListedColormap(['#000000', '#F0F3BD', '#76C893'])
        bounds = [-2, 0, 0.5, 300]
        norm = colors.BoundaryNorm(bounds, cmap.N)

        # Plot the heatmap
        plt.close('all')
        fig, ax = plt.subplots()
        im = ax.imshow(data_ep, cmap=cmap, norm=norm, **kwargs)

        # Show all ticks and label them with the respective list entries.
        ax.xaxis.set_major_locator(MultipleLocator(5))
        ax.xaxis.set_minor_locator(MultipleLocator(1))
        ax.yaxis.set_major_locator(MultipleLocator(5))
        ax.yaxis.set_minor_locator(MultipleLocator(1))

        # Turn spines off and create grid.
        ax.spines[:].set_visible(False)
        if step == -2:
            step = 'Final'
        ax.set_title(f"EP {ep} - {step} Step. Layout")
        ax.set_xticks(np.arange(data_ep.shape[1]+1)-.5, minor=True)
        ax.set_yticks(np.arange(data_ep.shape[0]+1)-.5, minor=True)
        ax.grid(which="minor", color="k", linestyle='-', linewidth=1)
        ax.tick_params(which="minor", bottom=False, left=False)
        # ax.legend()
        fig.tight_layout()

    def plot_1ep(self, ep=None):
        if ep == None:
            ep = self.default_ep
        data_ep = self.data[f"episode_{ep}"][:-1]
        rewards, actions = [], {0:0, 1:0, 2:0, 3:0, 4:0, 5:0, 6:0, 7:0}
        for step, info in enumerate(data_ep):
            rewards.append([int(step+1),float(info["reward"])])
            actions[info["action"]]+=1
        rewards = np.array(rewards)

        step = rewards[:,0]
        f_reward = rewards[:,1]

        plt.close('all')
        fig, ax = plt.subplots()
        ax.plot(step, f_reward, label='Reward', c='royalblue')

        ax.set(xlabel='step', title=f'EP {ep}. Convergence over steps')
        ax.legend(bbox_to_anchor=(1.05, 0.2),loc='upper left')
        ax.grid()

        print('step length:', len(data_ep))
        print("Distribution of Actions:",actions)
        
        plt.show()

        return

    def plot_eps(self, eps):
        plt.close('all')
        fig, ax = plt.subplots()
        for i, ep in enumerate(eps):
            data_ep = self.data[f"episode_{ep}"][:-1]
            reward = []
            for info in data_ep:
                reward.append(float(info["reward"]))
            ax.plot(np.array(range(self.max_steps)), np.array(reward), label=f'EP {ep}', c=self.c_chart[i%6])
        ax.set(xlabel='step', title='Convergence by Episodes')
        ax.legend(bbox_to_anchor=(1.05, 0.4),loc='upper left')
        ax.grid()
        plt.show()
        return

    def plot_eps_actions(self, eps):
        action_counts = {a:np.zeros(len(eps)) for a in range(8)}
        rewards_eps = {}
        for epi, ep in enumerate(eps):
            data_ep = self.data[f"episode_{ep}"][:-1]
            for info in data_ep:    # info of each step
                action_counts[info["action"]][epi] += 1
            rewards_eps[ep] = self.data[f"episode_{ep}"][-1]['avg reward']
        episodes = [f"EP {e}.\n(R: {round(rewards_eps[e],3)})" for e in eps]

        plt.close('all')
        fig, ax = plt.subplots()
        width = 0.5
        bottom = np.zeros(len(eps))
        for attr, action_count in action_counts.items():
            p = ax.bar(episodes, action_count, width, label=f"Action-{attr}", bottom=bottom, color=self.c_chart[attr])
            bottom += action_count
        ax.set(ylabel='Counts', title='Action Counts by Episode')
        ax.legend(bbox_to_anchor=(1.05,0.5),loc='upper left', reverse=True)
        plt.show()
        
        return

