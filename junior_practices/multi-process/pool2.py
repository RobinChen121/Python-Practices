"""
created on 2025/2/14, 11:34
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
from multiprocessing import Pool
import os

def worker(n):
    return f"Process {os.getpid()} computed {n * n}"

if __name__ == '__main__':
    # 使用 with 语句管理进程池
    with Pool(processes = 4) as pool:
        results = pool.map(worker, range(5))

    # 进程池自动关闭，不需要手动 pool.close() 和 pool.join()
    for res in results:
        print(res)