"""
created on 2025/2/3, 15:19
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
from multiprocessing import Process


class MyClass:
    para1 = 10
    para2 = 'chen'


    def foo1(self, arr):
        print(self.para1)
        print(arr)

    def foo2(self):
        print(self.para2)

    def run(self):
        arr = [[1, 2 ,3], [4, 5, 6]]
        K = 2
        procs = [Process(target=self.foo1, args=(arr[k], )) for k in range(K)]
        for k in range(K):
            procs[k] = Process(target=self.foo1, args=(arr[k],))
            procs[k].start()
        for k in range(K):
            procs[k].join()
