"""
created on 2025/2/12, 21:50
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
import pdb

def add(a, b):
    # pdb.set_trace()
    return a + b  # 如果在这里 "Step Out"，会直接执行 return 并返回到 main() 的调用处

def main():
    x = 10
    y = 20
    result = add(x, y)
    # pdb.set_trace()
    print(result)

main()