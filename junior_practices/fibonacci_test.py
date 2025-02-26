"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/22 22:30
Description: 
    

"""
import time

# Recursive Fibonacci
def fib_recursive(n):
    if n <= 1:
        return n
    else:
        return fib_recursive(n-1) + fib_recursive(n-2)

# Iterative Fibonacci
def fib_iterative(n):
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a

# Test for recursive Fibonacci
start_time = time.time()
fib_recursive(40)
end_time = time.time()
print(f"Python recursive Fibonacci time: {end_time - start_time} seconds")

# Test for iterative Fibonacci
start_time = time.time()
fib_iterative(30)
end_time = time.time()
print(f"Python iterative Fibonacci time: {end_time - start_time} seconds")