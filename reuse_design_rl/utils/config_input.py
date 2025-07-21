
import numpy as np
import random


def inv_configurator(input_shape_n, h=1, w=1):

    inv_config, i = {}, 0

    for shape_n in input_shape_n:   # shape_n = (cut_0/no-cut_1, size_X, size_Y, number)
        if shape_n[0]:  # no-cut == 1
            for _ in range(shape_n[-1]):
                inv_config[i] = stock_generator(shape_n[1], shape_n[2], h, w)
                i += 1
        else:
            for _ in range(shape_n[-1]):
                inv_config[i] = add_cuts(shape_n[1], shape_n[2], h, w)
                i += 1

    if inv_config == {}:
        inv_config = {0:np.array([0])}

    return inv_config


def stock_generator(size_x, size_y, h, w):
    stock = np.ones((size_y,size_x))
    return stock


def add_cuts(size_x, size_y, h, w):

    stock = stock_generator(size_x, size_y, h, w)
    rl = random.choice(range(1,3))
    cl = random.choice(range(1,4-rl))
    rci = [random.choice(range(i+1)) for i in np.array(stock.shape)-np.array([rl, cl])]
    stock[rci[0]:rci[0]+rl,rci[1]:rci[1]+cl] = 0

    return stock


def env_config(outer_shape={}, input_shape_n=[], MAX_STEPS=50):
    
    return {
        "outer_arr": np.zeros([outer_shape["h"], outer_shape["w"]]),
        "inventory": inv_configurator(input_shape_n, outer_shape["h"], outer_shape["w"]),
        "max_steps": MAX_STEPS
    }
