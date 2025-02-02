"""
created on 2025/2/1, 11:08
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
import multiprocessing
import os
import time

def print_records(records_):
    """
    function to print record(tuples) in records(list)
    """
    print('process id:', os.getpid())
    print('time is ', time.time())
    for record in records_:
        print("Name: {0}\nScore: {1}\n".format(record[0], record[1]))

def insert_record(record, records_):
    """
    function to add a new record to records(list)
    """
    print('process id:', os.getpid())
    print('time is ', time.time())
    records_.append(record)
    print("New record added!\n")

if __name__ == '__main__':
    with multiprocessing.Manager() as manager:
        # creating a list in server process memory
        records = manager.list([('Sam', 10), ('Adam', 9), ('Kevin',9)])
        # new record to be inserted in records
        new_record = ('Jeff', 8)

        # creating new processes
        p1 = multiprocessing.Process(target=insert_record, args=(new_record, records))
        p2 = multiprocessing.Process(target=print_records, args=(records,))

        # running process p1 to insert new record
        p1.start()
        p1.join()

        # running process p2 to print records
        p2.start()
        p2.join()
