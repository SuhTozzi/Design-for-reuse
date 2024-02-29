
import numpy as np
import random, math
import json
import os, glob, time

import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import pathfinding

__all__ = ["vizresult", "reorganize_path", "add_random2order", "assign_stocks", "sort_path_by_target"]

## Pathfinding
def reorganize_path(path_edges):
    paths_dir = []    
    for target_num, path_by_tar in enumerate(path_edges):
        path1_dir = []
        for i, (u,v) in enumerate(path_by_tar):
            temp_dir = np.array(list(v))-np.array(list(u))
            cur_dir = list((np.nonzero(temp_dir)[0]+1)*temp_dir[temp_dir != 0])[0]  # (nonzero_index)*(dir)
            if i==0:
                org_dir = cur_dir
                temp_len, order = 0, 0
            elif org_dir != cur_dir:
                path1_dir.append((org_dir, temp_len+1, 2, target_num, order))
                org_dir = cur_dir
                temp_len = 0
                order += 1
            else:
                temp_len += 1
        path1_dir.append((org_dir, temp_len+1, 2, target_num, order)) # Append the missing last edge
        paths_dir.append(path1_dir)
    return paths_dir

def add_random2order(path_org, random_prob, obs_nodes=[], opt=1):
    path_copy, i = path_org.copy(), 1                               # paths_dir_flat = (Dir_x1/y2/z3, Length, Reused:1/Undef:2, target#, order)
    if opt == 2: ## To re-order stocks again
        if len(path_copy) > 2:
            for curi, path in enumerate(path_copy[2:]):
                curi += 2
                dir_cur = path[0]
                if dir_cur == path_copy[curi-2][0]:
                    path_copy = check_obs(curi, curi-2, path_copy, obs_nodes)
                    continue
                if curi>=3 and dir_cur == path_copy[curi-3][0]:
                    path_copy = check_obs(curi, curi-3, path_copy, obs_nodes)

    elif opt == 3: ## To re-re-order stocks again
        while i < (1-random_prob)*len(path_org):
            rani = random.sample(range(len(path_copy)), k=1)[0]
            if rani < 1:
                path_copy = check_obs(rani, rani+1, path_copy, obs_nodes)
            else:
                path_copy = check_obs(rani, rani-1, path_copy, obs_nodes)
            i += 1
    return path_copy


def check_obs(curi, newi, path_copy, obs_nodes):
    path_copy = sorted(path_copy, key=lambda x: x[4])
    node_temp = [0,0,0]
    if curi == 0:   # (e.g., curi=0, newi=1)
        for index in [newi, curi]:
            node_temp_org = node_temp.copy()
            dir1, len1 = path_copy[index][:2]
            node_temp[abs(dir1)-1] = len1*dir1/abs(dir1)
            for obs1 in obs_nodes:
                if math.prod([len(set(range(int(node_temp_org[xyzi]), int(node_temp[xyzi]+1))).intersection(obs1[xyzi])) for xyzi in range(3)]) != 0:
                    return path_copy
        temp_cur = (path_copy[newi][0], path_copy[newi][1], path_copy[newi][2], path_copy[newi][3], path_copy[curi][4]+0.01)
        temp_new = (path_copy[curi][0], path_copy[curi][1], path_copy[curi][2], path_copy[curi][3], path_copy[newi][4]+0.01)
        path_copy[curi], path_copy[newi] = temp_cur, temp_new # Switch
        return path_copy

    else:
        ## Pre-processing before the 1st switch
        for path in path_copy[:newi+1]:
            dir1, len1 = path[:2]
            node_temp[abs(dir1)-1] += len1*dir1/abs(dir1)
        ## Test the 1st switch
        node_temp_org = node_temp.copy()
        dir2, len2 = path_copy[curi][:2]
        node_temp[abs(dir2)-1] += len2*dir2/abs(dir2)
        for obs1 in obs_nodes:
            if math.prod([len(set(range(int(node_temp_org[xyzi]), int(node_temp[xyzi]+1))).intersection(obs1[xyzi])) for xyzi in range(3)]) != 0:
                return path_copy      
        ## Test between the switches
        for path in path_copy[newi+1:curi]:
            node_temp_org = node_temp.copy()
            dir3, len3 = path[:2]
            node_temp[abs(dir3)-1] += len3*dir3/abs(dir3)
            for obs1 in obs_nodes:
                if math.prod([len(set(range(int(node_temp_org[xyzi]), int(node_temp[xyzi]+1))).intersection(obs1[xyzi])) for xyzi in range(3)]) != 0:
                    return path_copy
        ## Finally switch the values if necessary
        temp = (path_copy[curi][0], path_copy[curi][1], path_copy[curi][2], path_copy[curi][3], path_copy[newi][4]+0.001)
        if curi > newi:
            del path_copy[curi]
            path_copy.insert(newi+1, temp)
            return path_copy
        else:
            del path_copy[curi]
            path_copy.insert(newi, temp)
            return path_copy


def assign_stocks(paths_dir_flat, rem_stocks):
    rem_paths_dir = sorted(paths_dir_flat, key=lambda x: x[1], reverse=True)
    new_paths_dir = []
    rem_stocks1 = rem_stocks.copy()
    for stock in sorted(rem_stocks1, reverse=True):
        for path in rem_paths_dir:
            if stock == path[1]:
                new_paths_dir.append((path[0], stock, 1, path[3], path[4]))
                rem_paths_dir.remove(path)
                rem_stocks1.remove(stock)
                break
            elif stock < path[1]:
                temp_path = (path[0], path[1]-stock, path[2], path[3], path[4]+0.1)
                new_paths_dir.append((path[0], stock, 1, path[3], path[4]))
                rem_paths_dir.remove(path)
                rem_paths_dir.append(temp_path)
                rem_stocks1.remove(stock)
                break
    new_paths_dir.extend(rem_paths_dir)
    return new_paths_dir
    # return new_paths_dir, rem_paths_dir


def sort_path_by_target(paths_dir, paths_dir2, temp):
    for path in paths_dir: # paths_dir2 = {target_num: [path1, path2, ], ...}
        if path[3] not in temp:
            paths_dir2[path[3]] = [path]
            temp.append(path[3])
        else:
            paths_dir2[path[3]].append(path)
            paths_dir2[path[3]] = sorted(paths_dir2[path[3]], key=lambda x: x[4])
    return paths_dir2, temp



## Visualization
def vizresult(G, G_obs, targets, obstacles, result, start_time, dict4export):
    # Prepare for the .txt export
    dict4export["Execution time"] = str(round((time.time()-start_time)/60, 2))+" minutes"
    filename = f"./result/result"
    filelist = glob.glob(filename+"*.json")
    if len(filelist) == 0:
        fileindex = 1
    else:
        fileindex = max([int(f.split(".")[-2][-3:]) for f in filelist])+1
    
    # Pre-process environment and paths
    node_xyz_obs = np.array(G_obs)
    s_ts = targets.copy()
    s_ts.append((0,0,0))
    node_xyz_st = np.array(s_ts)

    solnum = len(result)
    tarnum = len(targets)+1
    ratio = (targets[0][0], targets[0][1], targets[0][2])

    fig = plt.figure(figsize=(15,15))

    # Pre-process xyz-ticks
    xticks, yticks, zticks = [0,ratio[0]], [ratio[1]], [ratio[2]]
    for obs in obstacles:
        x, y, z, w, h, d = obs    # lower top left corner
        xticks.extend([x, x+w])
        yticks.extend([y, y-h])
        zticks.extend([z, z+d])
    xticks, yticks, zticks = list(set(xticks)), list(set(yticks)), list(set(zticks))

    # Create figures
    for soli in range(solnum):
        # result is a pd.DataFrame w/3 columns:'index','score','path'
        paths, score_each, score = result[soli]
        dict4export["Final"][f'Solution{soli+1}'] = {'Fitness':score, 'Fitness_Length, Bending, Reuse, Parallel': score_each}
        re_final = []

        # Draw the integrated plot
        ax = Axes3D(fig)
        ax = fig.add_subplot(solnum, tarnum, soli*tarnum+1, projection="3d")
        ax.set_box_aspect(ratio)
        ax.scatter(*node_xyz_obs.T, s=50, c="k", zorder=2)
        ax.scatter(*node_xyz_st.T, s=50, c="r", marker='^', zorder=2)
        ax.set_title(f'{soli+1}:  score {score}', loc='left')
        ax.set_xticks(xticks)
        ax.set_yticks(yticks)
        ax.set_zticks(zticks)

        for tari, path_by_tar in enumerate(paths):
            # Draw basic pts and edges
            ax2 = fig.add_subplot(solnum, tarnum, soli*tarnum+2+tari, projection="3d")
            ax2.set_box_aspect(ratio)
            ax2.scatter(*node_xyz_obs.T, s=50, c="k", zorder=2)
            ax2.scatter(*node_xyz_st.T, s=50, c="r", marker='^', zorder=2)
            ax2.set_title(f'target {tari+1}', y=-0.1)
            # Draw edges: Path
            pts, re_edges, non_re = stock2path(path_by_tar)
            ax2.scatter(*np.array(pts[1:-1]).T, s=30, c="r", zorder=2)

            for vizedge in re_edges:
                ax2.plot(*np.array(vizedge).T, color="tab:blue", zorder=1)
                ax.plot(*np.array(vizedge).T, color="tab:blue", zorder=1)
            for vizedge in non_re:
                ax2.plot(*np.array(vizedge).T, color="tab:red", zorder=1)
                ax.plot(*np.array(vizedge).T, color="tab:red", zorder=1)
            ax2.set_xticks([x[0] for x in pts])
            ax2.set_yticks([x[1] for x in pts])
            ax2.set_zticks([x[2] for x in pts])

            # Stacking the results for export
            dict4export["Final"][f'Solution{soli+1}'][f'Target{tari+1}_pts'] = str(pts)
            re_final.append(re_edges)
        dict4export["Final"][f'Solution{soli+1}'][f'Reused_Stocks'] = str(re_edges)
        dict4export["Final"][f'Solution{soli+1}']['Raw_path'] = str(paths)

    # Finally export to json file
    if not os.path.exists("/".join(filename.split("/")[:-1])):
        os.makedirs("/".join(filename.split("/")[:-1]))
    with open(f'{filename}{fileindex:03d}.json', 'w+') as f:
        json.dump(dict4export, f, indent = 6)
    print(f"Results are exported to 'result{fileindex:03d}.json'")

    # Show the funal figure
    fig.tight_layout(pad=3)
    plt.savefig(f'{filename}{fileindex:03d}.png')
    plt.show()
    return


def stock2path(path_by_tar):
    pts, re_edges, non_re = [[0,0,0]], [], []
    for stock in path_by_tar:
        pt = pts[-1].copy()
        if stock[0] > 0:
            # (Direction_x1/y2/z3, Length, Reused:1/Undefined:2, target_num, order)
            pt[stock[0]-1] += stock[1]
        else:
            pt[-stock[0]-1] -= stock[1]
        pts.append(pt)
        if stock[2] == 1:
            re_edges.append(pts[-2:])
        else:
            non_re.append(pts[-2:])
    return pts, re_edges, non_re

