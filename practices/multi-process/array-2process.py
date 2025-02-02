"""
created on 2025/2/2, 20:52
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
from multiprocessing import Process

def process_part(arr, start, end):
    """处理数组的部分数据"""
    for i in range(start, end):
        arr[i] *= 2  # 这里假设是简单的数值翻倍操作
    print(f"处理部分数组: {arr[start:end]}")

if __name__ == "__main__":
    data = [1, 2, 3, 4, 5, 6, 7, 8]  # 要处理的数组
    n = len(data) // 2  # 计算分割点

    # 启动两个进程，分别处理数组的一半
    p1 = Process(target=process_part, args=(data, 0, n))
    p2 = Process(target=process_part, args=(data, n, len(data)))

    p1.start()
    p2.start()

    p1.join()
    p2.join()

    print("主进程最终数组:", data)  # 这不会正确更新！
