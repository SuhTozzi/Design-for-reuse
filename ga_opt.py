"""
This is a collection of functions for a genetic algorithm-based path and stock assignment optimization
"""
import numpy as np
import time
import random
import math

import pathfinding
import tools

__all__ = ["main"]

def main(targets, obstacles, stocks, num_generations, num_population, num_parents_mating, random_prob, num_new_p, weights):
    term_iter, best_fitness_score = 0, 0
    weights = np.array(weights)
    start_time = time.time()
    dict4export = {'Obstacles': str(obstacles), 'Targets': str(targets), "Stocks": str(stocks),
                   "Number of generations": num_generations, "Number of population": num_population, "Number of parents for mating": num_parents_mating,
                   "Random probability": random_prob, "Weights_Length, Bending, Reuse, Parallel": str(weights), "Execution time": 0,
                   "Final":{}, "Checkpoints":{}}

    # 0. Initialization
    G, G_obs, paths_dir, path_edges = initialization(targets, obstacles)
    offsprings = [paths_dir] # paths_dir = (Direction_x1/y2/z3, Length, Reused:1/Undefined:2, target_num, order)
    rem_stocks, archive = stocks.copy(), [] # Archive = [[paths_dir, fitness_each, fitness], [,], ...]
    
    for i in range(1, num_generations+1):
        # 1. Fitness function
        for paths_dir in offsprings:
            fitness, fitness_each = fitness_func(paths_dir, weights) # fitness_each = np.array(paths_len, num_bend, num_stocks, parallel_coeff)
            archive.append([paths_dir, fitness_each, fitness])
        ## Termination criteria
        ##1) Reached the target fitness value?
            if fitness > 99.9:
                print("Early stopping: Reached the target fitness value")
                break
        ##2) Reached the max generation?
        if i == num_generations:
            break
        ##3) Is the sum of the deviations bigger than the threshold?
        archive = sorted(archive, key=lambda x:x[-1], reverse=True)
        # documentation
        if i == 1 or i%100 or best_fitness_score < archive[0][-1] == 0:
            dict4export['Checkpoints'][f'Generation{i}']={'Fitness': archive[0][-1], 'Fitness_Length, Bending, Reuse, Parallel':archive[0][1], 'Path':str(archive[0][0])}

        best_fitness_score = archive[0][-1]
        if best_fitness_score - archive[-1][-1] == 0:
            # Early stopping
            if term_iter > 100:
                print("Early stopping: The deviation is too small")
                break
            else: term_iter += 1
        else:
            term_iter = 0

        # 2. Parent selection
        parents = archive[:int(num_parents_mating*(1-random_prob))]
        if int(num_parents_mating*(1-random_prob)) > int(num_parents_mating*random_prob*(1-num_new_p)):
            parents.append(random.sample(archive[int(num_parents_mating*(1-random_prob)):],int(num_parents_mating*random_prob*(1-num_new_p))))

        # 3. Mutate paths
        ## 3-1. Add 100% random paths
        n = 0
        while n < int(num_parents_mating*random_prob*num_new_p):
            _, _, paths_dir_mut, path_edges = initialization(targets, obstacles, path_edges)
            parents.append([paths_dir_mut, 0, 0])
            n += 1
        ## 3-2. Optimize the provided paths & Assign stocks
        offsprings = mating(parents, rem_stocks, num_population, random_prob, obstacles)

        # Communicating with users
        if i%100 == 0:
            print(f'Generation {i}/{num_generations} is completed --Best fitness score = {best_fitness_score}')

    # Finalize the optimization
    final_recommendations = sorted(archive, key=lambda x:x[-1], reverse=True)
    result = [final_recommendations[0]]
    path = final_recommendations[0][0]
    for x in final_recommendations[1:]:
        if x[0] == path:
            pass
        else:
            result.append(x)
            path = x[0]
        if len(result) == 3:
            break

    # Visualize the result
    print("All done! -- Preparing for visualization")
    tools.vizresult(G, G_obs, targets, obstacles, result, start_time, dict4export)

    return



def initialization(targets, obstacles, path_edges=0):
    G, G_obs = pathfinding.define_env(targets, obstacles)
    if path_edges != 0:
        for path_edge in path_edges:
            for u, v in path_edge:
                G.edges[tuple(u), tuple(v)]['weight'] = 10
    path_edges = pathfinding.find_paths(G, targets)
    paths_dir = tools.reorganize_path(path_edges)
    return G, G_obs, paths_dir, path_edges


def fitness_func(paths_dir, weights):
    # Analyze the paths
    paths_len, num_bend, num_stocks, parallel_coeff = 1, 1, 1, 1
    path0_np = clean_path4pp(paths_dir[0])
    for i, path in enumerate(paths_dir):
        num_bend += len(path)
        dir_old = path[0][0]
        for edge_info in path:
            paths_len += edge_info[1]
            if edge_info[2] == 1:
                num_stocks += edge_info[1]
            dir_new = edge_info[0]
            if dir_new != dir_old:
                num_bend += 5
                dir_old = dir_new

        if i != 0:
            path_np = clean_path4pp(path)
            indexing_std = min(len(path_np), len(path0_np))
            parallel_anal = list(np.array(path0_np[:indexing_std])-np.array(path_np[:indexing_std]))
            p_add = 0
            for j, x in enumerate(parallel_anal):
                if x[0] == 0 and p_add == 0:     # (Direction, Length, Reused, Target_num, Order)
                    parallel_coeff += min(paths_dir[0][j][1], path[j][1])
                    p_add = x[1]
                else:
                    break
    # Score the paths
    std_len = math.ceil(math.log10(paths_len))
    std_bend= math.ceil(math.log10(num_bend))
    std_prl = math.ceil(math.log10(parallel_coeff))
    score = np.array([(10-std_len-paths_len/(10**std_len))/2,
                      (10-std_bend-num_bend/(10**std_bend))/2,
                      (std_len+num_stocks/(10**std_len))*1.5,
                      (std_prl+parallel_coeff/(10**std_prl))*1.5])*10
    fitness_each = np.round(weights*score.T, 4)
    fitness = sum(fitness_each)
    return fitness, str(fitness_each)


def mating(parents, rem_stocks, num_population, random_prob, obstacles):
    obs_nodes = []
    for obs in obstacles: # (lower top left corner x, y, z, w, h, d)
        obs_nodes.append([range(obs[0], obs[0]+obs[3]+1),
                          range(obs[1]-obs[4], obs[1]+1),
                          range(obs[2], obs[2]+obs[5]+1)])
    offsprings = []
    for parent in parents:      # parent = [paths_dir, fitness_each, fitness]
        paths_dir = parent[0]   # paths_dir = (Direction_x1/y2/z3, Length, Reused:1/Undefined:2, target_num, order)
        paths_dir_flat = []
        for path in paths_dir:
            paths_dir_flat.extend(path)
        
        # Generate offsprings
        i = 0
        while i in range(num_population):
            i += 1
            # Assign stocks
            new_paths_dir = tools.assign_stocks(paths_dir_flat, rem_stocks)

            # Re-order the stocks
            paths_dir2, temp = tools.sort_path_by_target(new_paths_dir, {}, [])

            # Clean and add randomness to the re-ordered stocks for update
            p, p_updated, p_updated22 = [], [], [] # [[Paths4Target1], [P4Target2], ...]
            for k in sorted(temp):
                p_by_k = clean_path(paths_dir2[k])
                p.append(p_by_k)
                # Add random1
                p_by_k = tools.add_random2order(p_by_k, random_prob, obs_nodes, 2)
                p_by_k = clean_path(p_by_k)
                p_updated.append(p_by_k)
                # Add random2
                p_by_k = tools.add_random2order(p_by_k, random_prob, obs_nodes, 3)
                p_by_k = clean_path(p_by_k)
                p_updated22.append(p_by_k)
            
            # Update the offspring archive
            if p!=paths_dir:
                offsprings.append(p)
            if p_updated!=p and p_updated!=paths_dir:
                offsprings.append(p_updated)
            if p_updated22!=p and p_updated22!=p_updated and p_updated22!=paths_dir:
                offsprings.append(p_updated22)

    return offsprings


def clean_path(p_by_k):
    result, edge_old = [], (0,0,0)
    for i, edge in enumerate(p_by_k): # edge = (0_direction, 1_length, 2_reused?, 3_target#, 4_order)
        # Neighboring two edges
        if abs(edge[0])==abs(edge_old[0]) and edge[2]*edge_old[2]==4:
            new_len = edge_old[1] + edge[1]*edge[0]/abs(edge[0])
            if new_len == 0:
                result.pop()
                if i > 2: edge_old = p_by_k[i-2]
                else: edge_old = (0,0,0)
                continue
            edge = (int(edge_old[0]*new_len/abs(new_len)), int(abs(new_len)), edge_old[2], edge_old[3], edge_old[4])
            result.pop()
        result.append(edge)
        edge_old = edge
    result = sorted(result, key=lambda x: x[4])
    return result


def clean_path4pp(p_by_k): # Cleaning path to score Parallel Piping
    result, edge_old = [], (0,0,0)
    for edge in p_by_k: # edge = (0_direction, 1_length, 2_reused?, 3_target#, 4_order)
        # Neighboring two edges
        if abs(edge[0])==abs(edge_old[0]):
            new_len = edge_old[1] + edge[1]*edge[0]/abs(edge[0])
            edge = (int(edge_old[0]*new_len/abs(new_len)), int(abs(new_len)), edge_old[2], edge_old[3], edge_old[4])
            result.pop()
        result.append(edge)
        edge_old = edge
    return result

