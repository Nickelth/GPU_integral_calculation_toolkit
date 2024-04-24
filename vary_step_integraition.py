import contextlib
import sqlite3
import numpy as np
import thread
import math
from numba import  cuda
from time import time

def f(x):
    return 1 / (1 + x**4) + 23 * math.sin(x) - 7 / (2 * x**6 + 32) + 3 * x**4 + 21314 * math.cos(x)
    #return  x**3 + 1
@cuda.jit
def f_cuda(x):
    return 1 / (1 + x**4) + 23 * math.sin(x) - 7 / (2 * x**6 + 32) + 3 * x**4 + 21314 * math.cos(x)
    #return x**3 + 1
def trapezium_integration(l, r, step, piece):
    # print("trap start")
    res = 0;
    p_num = piece
    # print(step)
    # print(piece)
    for i in range(0, p_num):
        cur_l = l + i * step
        cur_r = cur_l + step
        res = res + f(cur_l) + f(cur_r)
        # print(cur_l)
        # print(cur_r)

    res = res * (step / 2)
    # print(res)
    # print("trap end")
    return res

def vary_step_integration(l, r, step, piece, last):
    res = 0
    for i in range(0, piece):
        if i % 2 == 1:
            res = res + f(l + step * i)
    res = last / 2 + (r - l) / (piece) * res
    return res

def normal_func(l, r, piece):

    step = (r - l) / piece

    res = 0
    start = time()
    eps = 1000
    cur = 0
    last = trapezium_integration(l, r, step, piece)
    #print(last)
    cnt = 0
    for i in range(0, 15):
        step = step / 2
        piece = piece * 2
        if(step <= 0):
            break
        cur = vary_step_integration(l, r, step, piece, last)
        #print(cur)
        if(math.fabs(cur - last) < eps):
            break
        last = cur

    print(cur)
    print("Normal function finished calculation in " + str(time() - start) + " seconds")

@cuda.jit
def trapezium_integration_gpu(last_device, l, r, step):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    cur_l = l + idx * step
    cur_r = cur_l + step
    last_device[idx] = f_cuda(cur_r) + f_cuda(cur_l)

@cuda.jit
def vary_step_integration_gpu(cur_device, l, r, step, N):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if(idx % 2 == 0 or idx >= N):
        return
    cur_device[idx] = f_cuda(l + idx * step)

def gpu_func(l, r, num_block, num_thread):
    start = time()
    tmp = np.zeros(num_block * num_thread, dtype=float)
    last_device = cuda.to_device(tmp)
    step = (r - l) / (num_block * num_thread)
    piece = (num_block * num_thread)

    eps = 1000
    trapezium_integration_gpu[num_block, num_thread](last_device, l, r, step)
    cuda.synchronize()
    last = np.sum(last_device.copy_to_host()) * (step / 2)
    #print(last)
    for i in range(0, 15):
        step = step / 2
        piece = piece * 2
        if (step <= 0):
            break
        num_block = piece // num_thread
        if(piece % num_thread != 0):
            num_block = num_block + 1
        cur_device = cuda.to_device(np.zeros(piece, dtype=float))
        vary_step_integration_gpu[num_block, num_thread](cur_device, l, r, step, piece)
        cuda.synchronize()
        sum = np.sum(cur_device.copy_to_host())
        cur = last / 2 + (r - l) / (piece) * sum
        #print(cur)
        if (math.fabs(cur - last) < eps):
            break;
        last = cur
    print(cur)
    print("GPU function finished calculation in " + str(time() - start) + " seconds")
if __name__ == "__main__":
    l = -20000
    r = 20000
    num_block = 8
    num_thread = 32
    normal_func(l, r, num_block * num_thread)
    gpu_func(l, r, num_block, num_thread)






