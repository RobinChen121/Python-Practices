"""
created on 2025/2/1, 13:19
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

Pipes : A pipe can have only two endpoints. Hence, it is preferred over queue
when only two-way communication is required.
multiprocessing module provides Pipe() function which returns
a pair of connection objects connected by a pipe.
The two connection objects returned by Pipe() represent the two ends of the pipe.
Each connection object has send() and recv() methods (among others).

"""
import multiprocessing
import os
import time

def sender(conn, msgs):
    """
    function to send messages to other end of pipe
    """
    print('process id:', os.getpid())
    print('time is ', time.time())
    for msg in msgs:
        conn.send(msg)
        print("Sent the message: {}".format(msg))
    conn.close()

def receiver(conn):
    """
    function to print the messages received from other
    end of pipe
    """
    print('process id:', os.getpid())
    print('time is ', time.time())
    while 1:
        msg = conn.recv()
        if msg == "END":
            break
        print("Received the message: {}".format(msg))

if __name__ == "__main__":
    # messages to be sent
    msgs = ["hello", "hey", "hru?", "END"]

    # creating a pipe
    parent_conn, child_conn = multiprocessing.Pipe()

    # creating new processes
    p1 = multiprocessing.Process(target=sender, args=(parent_conn,msgs))
    p2 = multiprocessing.Process(target=receiver, args=(child_conn,))

    # running processes
    p1.start()
    p2.start()

    # wait until processes finish
    p1.join()
    p2.join()
