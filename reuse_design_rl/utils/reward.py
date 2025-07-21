
import numpy as np
import math

def velocity_concav(canvas, stock, ref_yx):
    # Less concavity
    step_reward = 0
    stock_h, stock_w = np.shape(stock)
    canvas_temp = canvas.copy()
    canvas_temp[ref_yx[0]:ref_yx[0]+stock_h, 
                ref_yx[1]:ref_yx[1]+stock_w] += 1000
    
    vertex_par = np.array([0, np.shape(canvas)[1]-1])
    for r in range(canvas_temp.shape[0]):
        if np.any(canvas_temp[r,:] >= 100):
            c = np.argwhere(canvas_temp[r,:] >= 100)
            vertex_cur = np.array([c.min(), c.max()])
            if len(np.unique(c)) == 1 or list(np.unique(c)) != list(range(c.min(), c.max()+1)):
                step_reward -= np.unique(c).size    # Partially empty in the middle
            if c.min() != 0:
                step_reward -= 1  # Empty first column
            step_reward -= np.count_nonzero(vertex_par!=vertex_cur)*2     # Min corners
            vertex_par = vertex_cur
        elif not np.any(canvas_temp[r,:] >= 100) and step_reward != 0:
            step_reward -= 1  # Row fully empty in the middle
    step_reward = round(2*step_reward/(4*np.shape(canvas_temp)[0]+np.shape(canvas_temp)[1]), 3)   # More reuse with less shape complexity

    return step_reward

def velocity(canvas, stock, ref_yx):

    step_reward = 0
    canvas_h, canvas_w = np.shape(canvas)
    canvas_area = np.prod([canvas_w, canvas_h])

    # Possible geometric overlap
    if True in (ref_yx<0) or (ref_yx[0]+np.shape(stock)[0] > canvas_h) or (ref_yx[1]+np.shape(stock)[1] > canvas_w):
        step_reward -= 2    # Out-of-range
        
    # Possible reuse
    step_reward += (round(np.count_nonzero(stock)/(canvas_area), 3)-1)

    return step_reward


def shape(stock):

    return np.count_nonzero(stock)


def translate(stock, ref_yx, canvas, w=1, h=1):

    step_reward = 0
    stock_h, stock_w = stock.shape

    if (ref_yx[0]+stock_h > h) or (ref_yx[1]+stock_w > w):
        step_reward = 10
    else:
        stock_temp = stock + canvas[ref_yx[0]:ref_yx[0]+stock_h,
                                    ref_yx[1]:ref_yx[1]+stock_w]
        if True in (stock_temp > 1):
            step_reward = 10

    return step_reward


def environment(stock):

    return np.count_nonzero(stock==1)*3


def concavity(canvas, w=1):
    
    step_reward = 0
    vertex_par = np.array([0, w-1])

    for r in range(canvas.shape[0]):
        if np.any(canvas[r,:] >= 100):
            c = np.argwhere(canvas[r,:] >= 100)
            vertex_cur = np.array([c.min(), c.max()])
            if len(np.unique(c)) == 1 or list(np.unique(c)) != list(range(c.min(), c.max()+1)):
                step_reward += np.unique(c).size    # Missing the middle part
            if c.min() != 0:
                step_reward += 1
            step_reward += np.count_nonzero(vertex_par!=vertex_cur)  # Min corners
            vertex_par = vertex_cur
        elif not np.any(canvas[r,:] >= 100) and step_reward != 0:
            step_reward += 1

    return step_reward*3


def normalize(reward, w=1, h=1):

    return round((10**(math.log(w*h,10)//1-2))*reward/(w*h),4)

