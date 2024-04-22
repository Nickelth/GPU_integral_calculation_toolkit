import contextlib
import sqlite3

import numpy as np
import thread
import math
from numba import  cuda
from time import time



def f(x):
    #return 1 / (1 + x**4) + 23 * math.sin(x) - 7 / (2 * x**6 + 32) + 3 * x**4 + 21314 * math.cos(x)
    return  x**2 + 1
@cuda.jit
def f_cuda(x):
    #return 1 / (1 + x**4) + 23 * math.sin(x) - 7 / (2 * x**6 + 32) + 3 * x**4 + 21314 * math.cos(x)
    return x**2 + 1

def trapezium_integration(l, r, step, piece):
    res = 0;
    p_num = piece

    for i in range(0, p_num):
        cur_l = l + i * step
        cur_r = cur_l + step
        res = res + f(cur_l) + f(cur_r)
        # if(i == 0):
            # print("normal")
            # print(cur_l)
            # print(cur_r)
            # print(f(cur_l))
            # print(f(cur_r))
            # print(f(cur_l) + f(cur_r))
    res = res * (step / 2)
    return res
def vary_step_integration(l, r, step, last):
    res = 0
    for i in range(0, (r - l) // step):
        if i % 2 == 1:
            res = res + f(l + step * i)
    res = last / 2 + (r - l) / ((r - l) / step) * res
    return res

def normal_func(l, r, piece):

    step = (r - l) // piece

    res = 0
    start = time()
    eps = 0.001
    cur = 0
    T = []
    last = trapezium_integration(l, r, step, piece)
    T.append(last)
    while(1):
        step = step // 2
        if(step <= 0):
            break
        cur = vary_step_integration(l, r, step, last)
        print(cur)
        T.append(cur)
        if(math.fabs(cur - last) < eps):
            break
        last = cur
    if(len(T) < 4):
        print("Length of T must be at least 4!")
        return
    # print(cur)
    C = []
    P = 4
    m = P
    for i in range(0, 3):
        for j  in range(0, len(T) - 1):

            C.append((m * T[j + 1] - T[j]) / (m - 1))
            m = m * P

        T = C.copy()
        C = []
    print(T[-1])
    print("Normal function finished calculation in " + str(time() - start) + " seconds")
@cuda.jit
def trapezium_integration_gpu(last_device, l, r, step):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x


    cur_l = l + idx * step
    cur_r = cur_l + step
    last_device[idx] = f_cuda(cur_r) + f_cuda(cur_l)
    # if(idx == 0):
    #     print(cur_l)
    #     print(cur_r)
    #     print(f_cuda(cur_l))
    #     print(f_cuda(cur_r))
    #     print(last_device[idx])

@cuda.jit
def vary_step_integration_gpu(cur_device, l, r, step, N):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if(idx % 2 == 0 or idx >= N):
        return
    #print(l + idx * step)
    cur_device[idx] = f_cuda(l + idx * step)
    #cuda.atomic.add(cur_device[0], f_cuda(l + idx * step))
def get_num_block(piece, num_thread):
    num_block = piece // num_thread
    if (piece % num_thread != 0):
        num_block = num_block + 1
    return num_block
@cuda.jit
def romberg(res, T_cuda, N):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if idx > N:
        return
    m = pow(4, idx + 1)
    res[idx] = (m * T_cuda[idx + 1] - T_cuda[idx]) / (m - 1)
def gpu_func(l, r, num_block, num_thread):
    start = time()

    tmp = np.zeros(num_block * num_thread, dtype=float)
    last_device = cuda.to_device(tmp)
    step = (r - l) // (num_block * num_thread)

    start = time()
    eps = 0.001

    trapezium_integration_gpu[num_block, num_thread](last_device, l, r, step)
    cuda.synchronize()

    last = np.sum(last_device.copy_to_host()) * (step / 2)
    T = []
    #print(last)
    T.append(last)
    while (1):
        step = step // 2
        if (step <= 0):
            break

        num_block = get_num_block((r - l) // step, num_thread)
        cur_device = cuda.to_device(np.zeros(num_block * num_thread, dtype=float))

        vary_step_integration_gpu[num_block, num_thread](cur_device, l, r, step, num_block * num_thread)
        cuda.synchronize()
        sum = np.sum(cur_device.copy_to_host())
        cur = last / 2 + (r - l) / ((r - l) / step) * sum
        T.append(cur)
        if (math.fabs(cur - last) < eps):
            break;
        last = cur
    # print(cur)

    for i in range(0, 3):
        num_block = get_num_block(len(T) - 1, num_thread)
        tmp = cuda.to_device(np.zeros(num_block * num_thread, dtype=float))
        T_cuda = cuda.to_device(T)
        romberg[num_block, num_thread](tmp, T_cuda, len(T_cuda) - 1)
        T = T_cuda.copy_to_host()

    print(T[-1])
    print("GPU function finished calculation in " + str(time() - start) + " seconds")
if __name__ == "__main__":
    # compute integration of func() in [a, b]
    l = -1000000
    r = 1000000
    num_block = 4
    num_thread = 32
    normal_func(l, r, num_block * num_thread)
    gpu_func(l, r, num_block, num_thread)







