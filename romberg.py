import numpy as np
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
    res = 0;
    p_num = piece

    for i in range(0, p_num):
        cur_l = l + i * step
        cur_r = cur_l + step
        res = res + f(cur_l) + f(cur_r)


    res = res * (step / 2)

    return res
def vary_step_integration(l, r, step, last, piece):
    res = 0
    # print("vary start")

    for i in range(0, piece):
        if i % 2 == 1:
            res = res + f(l + step * i)
    #         print(l + step * i)
    # print("vary end")
    res = last / 2 + (r - l) / (piece) * res
    return res
def normal_func(l, r, piece):

    step = (r - l) / piece
    # print(step)
    # print(piece)
    res = 0
    start = time()
    eps = 1
    cur = 0
    T = []
    last = trapezium_integration(l, r, step, piece)
    T.append(last)
    #print(last)
    cnt = 0
    for i in range(0, 15):
        step = step / 2
        piece = piece * 2
        if(step <= 0):
            break
        cur = vary_step_integration(l, r, step, last, piece)
        #print(cur)
        T.append(cur)
        cnt = cnt + 1
        if(math.fabs(cur - last) < eps and cnt > 4):
            break
        last = cur
    if(len(T) < 4):
        print("Length of T must be at least 4!")
        return
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

@cuda.jit
def vary_step_integration_gpu(cur_device, l, r, step, N):
    idx = cuda.threadIdx.x + cuda.blockDim.x * cuda.blockIdx.x
    if(idx % 2 == 0 or idx >= N):
        return
    cur_device[idx] = f_cuda(l + idx * step)
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
    device_time = 0
    tmp = np.zeros(num_block * num_thread, dtype=float)
    device_st = time()
    last_device = cuda.to_device(tmp)
    device_time = device_time + time() - device_st
    step = (r - l) / (num_block * num_thread)
    piece = (num_block * num_thread)

    eps = 1

    trapezium_integration_gpu[num_block, num_thread](last_device, l, r, step)
    cuda.synchronize()
    device_st = time()
    last = np.sum(last_device.copy_to_host()) * (step / 2)
    device_time = device_time + time() - device_st
    #print(last)
    T = []

    T.append(last)
    cnt = 0
    for i in range(0, 15):
        step = step / 2
        if (step <= 0):
            break
        piece = piece * 2
        num_block = get_num_block(piece, num_thread)
        device_st = time()
        cur_device = cuda.to_device(np.zeros(piece, dtype=float))
        device_time = device_time + time() - device_st
        vary_step_integration_gpu[num_block, num_thread](cur_device, l, r, step, piece)
        cuda.synchronize()
        device_st = time()
        sum = np.sum(cur_device.copy_to_host())
        device_time = device_time + time() - device_st
        cur = last / 2 + (r - l) / (piece) * sum
        #print(cur)
        T.append(cur)
        cnt = cnt + 1
        if (math.fabs(cur - last) < eps and cnt > 4):
            break;
        last = cur
    for i in range(0, 3):
        num_block = get_num_block(len(T) - 1, num_thread)
        device_st = time()
        tmp = cuda.to_device(np.zeros(num_block * num_thread, dtype=float))
        T_cuda = cuda.to_device(T)
        device_time = device_time + time() - device_st
        romberg[num_block, num_thread](tmp, T_cuda, len(T_cuda) - 1)
        device_st = time()
        T = T_cuda.copy_to_host()
        device_time = device_time + time() - device_st
    print(T[-1])
    print("GPU function finished calculation in " + str(time() - start) + " seconds")
    print(str(device_time) + " seconds cost to transfer data.")
if __name__ == "__main__":
    l = -20000
    r = 20000
    num_block = 16
    num_thread = 128
    normal_func(l, r, num_block * num_thread)
    gpu_func(l, r, num_block, num_thread)







