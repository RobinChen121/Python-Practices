"""
created on 2025/2/2, 16:11
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""

from multiprocessing import Process
import pickle

class Worker:
    def do_work(self):
        print("Working")

if __name__ == '__main__':
    w = Worker()
    pickle.dumps(w.do_work())
    p = Process(target=w.do_work())
    p.start()
    p.join()

# from multiprocessing import Process
#
# class Worker:
#     @staticmethod
#     def do_work(self):
#         print("Working")
#
# if __name__ == 'main':
#     w = Worker()
#     p = Process(target=w.do_work)
#     p.start()
#     p.join()

# from multiprocessing import Process
#
# class MyClass:
#     def run(self):
#         print("Running")
#
# if __name__ == 'main':
#     obj = MyClass()
#     p = Process(target=obj.run)
#     p.start()
#     p.join()
