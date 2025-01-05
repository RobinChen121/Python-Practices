# -*- coding: utf-8 -*-
"""
Created on Sun Jan  5 13:48:16 2025

@author: chen
"""
from multiprocessing import Process, Queue, Pipe, Lock

def worker(q, lock):
    with lock:
        print("Task executed")
        q.put("Data")

if __name__ == "__main__":
    q = Queue()
    lock = Lock()
    p = Process(target=worker, args=(q, lock))
    p.start()
    p.join()
    print(q.get())  # Output "Data"