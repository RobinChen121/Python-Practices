"""
created on 2025/2/1, 17:14
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

lock = multiprocessing.Lock() 创建一个锁对象。
with lock: 语法确保 acquire() 和 release() 过程自动完成，避免死锁。
p1 和 p2 竞争修改 counter，但由于 Lock，每次只有一个进程能修改它。

"""
import multiprocessing
import time
import os

# 共享变量
counter = 0

# 进程安全地修改变量
def increment(lock):
    global counter
    for _ in range(5):
        print('process id:', os.getpid())
        print('time is ', time.time())
        with lock:  # 自动 acquire() 和 release()
            temp = counter
            time.sleep(0.1)  # 模拟计算时间
            counter = temp + 1
            print(f"Counter: {counter}")

if __name__ == "__main__":
    lock = multiprocessing.Lock()
    p1 = multiprocessing.Process(target=increment, args=(lock,))
    p2 = multiprocessing.Process(target=increment, args=(lock,))

    p1.start()
    p2.start()
    p1.join()
    p2.join()

    print(f"Final counter: {counter}")
