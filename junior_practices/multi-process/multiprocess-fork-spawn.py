"""
created on 2025/2/2, 15:39
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

直接复制当前进程（fork） 和 启动一个全新的 Python 解释器进程（spawn） 的核心区别在于 子进程的起始状态。简单来说：

fork：子进程从父进程的当前状态继续执行，所有的内存、变量、代码执行位置都会被复制。
spawn：子进程从零开始，类似于新运行 python script.py，不继承父进程的变量和执行状态。

如果一个 Python 进程先用了 fork，然后又用了 spawn（或反过来），可能会导致 崩溃、死锁或不可预测的行为。主要原因包括：

Python 只允许设置一次 multiprocessing 的启动方式
fork 和 spawn 进程的工作方式不同
混用可能导致资源（锁、文件句柄、线程）的问题
"""

from multiprocessing import Process
import os

def worker():
    print(f"子进程 {os.getpid()}，父进程 {os.getppid()}")

if __name__ == "__main__":
    import multiprocessing
    multiprocessing.set_start_method('spawn')  # 设置为 spawn 方式
    p = Process(target=worker)
    p.start()
    p.join()


# from multiprocessing import Process
# import os
#
# def worker():
#     print(f"子进程 {os.getpid()}，父进程 {os.getppid()}")
#
# if __name__ == "__main__":
#     import multiprocessing
#     multiprocessing.set_start_method('fork')  # 设置为 fork 方式
#     p = Process(target=worker)
#     p.start()
#     p.join()

