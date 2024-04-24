import contextlib
import sqlite3
import numpy as np
import thread
import math
from numba import  cuda
from time import time

def f(x):
    return 1 / (1 + x**4) + 23 * math.sin(x) - 7 / (2 * x**6 + 32) + 3 * x**4 + 21314 * math.cos(x)
@cuda.jit
def f_cuda(x):
    return 1 / (1 + x**4) + 23 * math.sin(x) - 7 / (2 * x**6 + 32) + 3 * x**4 + 21314 * math.cos(x)
def multi_Cotes(l, r):
    h = (r - l) / 4;
    res = (r - l) / 90 * (7 * f(l) + 32 * f(l + h) + 12 * f(l + 2 * h) + 32 * f(l + 3 * h) + 7 * f(r));
    return res
@cuda.jit
def gpu_multi_Cotes(tot_l, tot_r, step, res):

    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    l = step * idx + tot_l
    r = l + step
    h = (r - l) / 4;
    res[idx] = (r - l) / 90 * (7 * f_cuda(l) + 32 * f_cuda(l + h) + 12 * f_cuda(l + 2 * h) + 32 * f_cuda(l + 3 * h) + 7 * f_cuda(r));
if __name__ == "__main__":
    # compute integration of func() in [a, b]
    l = -20000
    r = 20000
    num_blocks = 16
    num_thread = 128
    h = (r - l) // (num_blocks * num_thread)
    res = 0
    start = time()
    for i in range(0, (r - l) // h):
        cur_l = l + i * h
        cur_r = cur_l + h
        res = res + multi_Cotes(cur_l, cur_r)
    print(res)
    print("Normal Cotes finished calculation in " + str(time() - start) + " seconds")
    start = time()

    res = np.zeros(num_thread * num_thread, dtype=float)
    res_device = cuda.to_device(res)
    step =  (r - l) // (num_thread * num_blocks)
    #start = time()
    gpu_multi_Cotes[num_blocks, num_thread](l, r, step, res_device)
    cuda.synchronize()
    res_host = res_device.copy_to_host()
    print(np.sum(res_host))
    print("GPU Cotes finished calculation in " + str(time() - start) + " seconds")



