"""Shortest paths and path lengths using the A* ("A star") algorithm.
"""
from heapq import heappop, heappush
from itertools import count

import networkx as nx
from networkx.algorithms.shortest_paths.weighted import _weight_function

import random

__all__ = ["define_env", "find_paths"]


def define_env(graph_dim, obstacles):
    # Define an initial graph G
    (x1, y1, z1) = graph_dim[0]
    G = nx.grid_graph(dim=(z1+1, y1+1, x1+1)) # It's wierd but should be in (y,x) order!
    # Add obstacle information to the initial graph G
    obstacle_nodes_integrated = []
    obstacle_nodes = obs_nodes(obstacles)
    obstacle_nodes_integrated.extend(obstacle_nodes)
    # obstacle_nodes = obs_nodes(obstacles)
    G_obs = G.subgraph(obstacle_nodes)
    G2 = add_nodes_cost(G, G_obs, 10000)
    return G2, G_obs


def find_paths(G, targets):
    path_edges = []
    for target in targets:
        result = astar_path(G, (0,0,0), target, heuristic=final_cost, weight='weight') # Returns a list of nodes
        path1 = []
        pair = [result[0]]
        for node in result[1:]:
            pair.append(node)
            path1.append(tuple(pair))
            pair = [node]
        path_edges.append(path1)
    return path_edges


def astar_path(G, source, target, heuristic=None, weight="weight"):
    if source not in G or target not in G:
        msg = f"Either source {source} or target {target} is not in G"
        raise nx.NodeNotFound(msg)

    if heuristic is None:
        # The default heuristic is h=0 - same as Dijkstra's algorithm
        def heuristic(u, v, z, w):
            return 0

    push = heappush
    pop = heappop
    weight = _weight_function(G, weight)

    G_succ = G._adj  # For speed-up (and works for both directed and undirected graphs)

    # The queue stores priority, node, cost to reach, and parent.
    # Uses Python heapq to keep in priority order.
    # Add a counter to the queue to prevent the underlying heap from
    # attempting to compare the nodes themselves. The hash breaks ties in the
    # priority and is guaranteed unique for all nodes in the graph.
    c = count()
    queue = [(0, next(c), source, 0, None)]

    # Maps enqueued nodes to distance of discovered paths and the
    # computed heuristics to target. We avoid computing the heuristics
    # more than once and inserting the node into the queue too many times.
    enqueued = {}
    # Maps explored nodes to parent closest to the source.
    explored = {}

    while queue:
        # Pop the smallest item from queue.
        _, __, curnode, dist, parent = pop(queue)

        if curnode == target:
            path = [curnode]
            node = parent
            while node is not None:
                path.append(node)
                node = explored[node]
            path.reverse()
            return path

        if curnode in explored:
            # Do not override the parent of starting node
            if explored[curnode] is None:
                continue

            # Skip bad paths that were enqueued before finding a better one
            qcost, h = enqueued[curnode]
            if qcost < dist:
                continue

        explored[curnode] = parent

        for neighbor, w in G_succ[curnode].items():
            cost = weight(curnode, neighbor, w)
            if cost is None:
                continue
            ncost = dist + cost
            if neighbor in enqueued:
                qcost, h = enqueued[neighbor]
                # if qcost <= ncost, a less costly path from the
                # neighbor to the source was already determined.
                # Therefore, we won't attempt to push this neighbor
                # to the queue
                if qcost <= ncost:
                    continue
            else:
                h = heuristic(neighbor, target, parent, curnode)
                h = h*ncost

            enqueued[neighbor] = ncost, h
            push(queue, (ncost + h, next(c), neighbor, ncost, curnode))

    raise nx.NetworkXNoPath(f"Node {target} not reachable from {source}")


def final_cost(neighbor, target, parent, curnode):
    if parent == None:
        parent = curnode
    (x1, y1, z1) = neighbor
    (x2, y2, z2) = target
    (x3, y3, z3) = parent
    (x4, y4, z4) = curnode

    direction_p2c = cal_direction([x4-x3, y4-y3, z4-z3])
    direction_c2n = cal_direction([x1-x4, y1-y4, z1-z4])
    direction_n2t = cal_direction([x2-x1, y2-y1, z2-z1])

    h = 1
    w = round(random.random(),3)
    if direction_p2c != direction_c2n:
        h += (3+w)*len(direction_n2t)
    elif direction_c2n[0] in direction_n2t:
        h += (1+w)*(len(direction_n2t)-1)
    else:
        h += (1+w)*len(direction_n2t)

    return h


## Helper functions
def obs_nodes(obstacles):
    result = []
    for obs in obstacles: # obs = (x, y, z, w, h, d): lower top left corner
        x, y, z, w, h, d = obs
        for i in range(w+1):
            for j in range(h+1):
                for k in range (d+1):
                    result.append((x+i, y-j, z+k))
    return result

def add_nodes_cost(G, G_obs, w):
    for (u, v) in list(G_obs.edges()):
        G.edges[u, v]['weight'] = w
        (x1, y1, z1) = u
        if x1 > 1:
            G.edges[(x1-1, y1, z1), (x1,y1, z1)]['weight'] = w
        if y1 > 1:
            G.edges[(x1, y1-1, z1), (x1,y1, z1)]['weight'] = w
        if z1 > 1:
            G.edges[(x1, y1, z1-1), (x1,y1, z1)]['weight'] = w
    return G

def cal_direction(xyz):
    my_direction = []
    for i, val in enumerate(xyz):
        if val != 0:
            my_direction.append(i)
    return my_direction
