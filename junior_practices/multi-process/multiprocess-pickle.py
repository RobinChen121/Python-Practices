"""
created on 2025/2/2, 15:09
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
# from multiprocessing import Process
#
# def print_value(x):
#     print(x)
#
# def start_process():
#     p = Process(target=print_value, args=(10,)) # 若是 lambda 函数，则会出现 pickle 错误
#     p.start()
#     p.join()
#
# # multiprocessing模块使用spawn方法启动子进程。这意味着子进程会重新导入主模块。
# # 如果主模块中没有使用if __name__ == '__main__':来保护进程启动代码，
# # 子进程会尝试重新执行主模块中的代码，从而导致递归创建新进程。
# # Python 中的 spawn 方式会创建一个全新的 Python 解释器进程，
# # 而不是 fork 现有进程（适用于 Windows 和 macOS）。
# if __name__ == '__main__':
#     start_process()

# 确保类定义在模块的全局作用域
class MyClass:
    def __init__(self, value):
        self.value = value

def worker(obj):
    return obj.value * 2

if __name__ == "__main__":
    from multiprocessing import Pool

    obj = MyClass(10)
    with Pool(2) as p:
        result = p.map(worker, [obj])
    print(result)
