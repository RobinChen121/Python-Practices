"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/25 15:07
Description: 
    

"""
import multiprocessing

def worker(queue, num):
    queue.put(num * num)  # 进程安全地存入队列

if __name__ == '__main__':
    queue = multiprocessing.Queue()  # 共享队列
    processes = []

    for i in range(5):
        p = multiprocessing.Process(target=worker, args=(queue, i))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    results = [queue.get() for _ in range(5)]
    print(results)