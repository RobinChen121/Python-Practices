"""
created on 2025/2/2, 13:09
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
from multiprocessing import Process, Value, Lock
import time
import os

def worker(lock, shared_value):
    print('process id:', os.getpid())
    print('time is ', time.time())
    for _ in range(1000):
        with lock:
            shared_value.value += 1
    print(shared_value.value)

if __name__ == '__main__':
    # 创建一个共享的整数值
    shared_value = Value('i', 0)

    # 创建一个锁对象
    lock = Lock()

    # 创建多个进程
    processes = []
    for _ in range(4):  # 4个进程
        p = Process(target=worker, args=(lock, shared_value))
        processes.append(p)
        p.start()

    # 等待所有进程完成
    for p in processes:
        p.join()

    print(f'Final value: {shared_value.value}')