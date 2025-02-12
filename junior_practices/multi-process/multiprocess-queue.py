"""
created on 2025/2/1, 11:59
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

Queue : A simple way to communicate between process with multiprocessing is
to use a Queue to pass messages back and forth.
Any Python object can pass through a Queue.

"""
import multiprocessing
import os
import time

def square_list(mylist, q):
    """
    function to square a given list
    """
    # append squares of mylist to queue
    print('process id:', os.getpid())
    print('time is ', time.time())
    for num in mylist:
        q.put(num * num)

def print_queue(q):
    """
    function to print queue elements
    """


    print('process id:', os.getpid())
    print('time is ', time.time())
    print("Queue elements:")
    while not q.empty():
        print(q.get())
    print("Queue is now empty!")

if __name__ == "__main__":
    # input list
    mylist = [1,2,3,4]

    # creating multiprocessing Queue
    q = multiprocessing.Queue()

    # creating new processes
    p1 = multiprocessing.Process(target=square_list, args=(mylist, q))
    p2 = multiprocessing.Process(target=print_queue, args=(q,))

    # running process p1 to square list
    p1.start()
    p1.join()

    # running process p2 to get queue elements
    p2.start()
    p2.join()
