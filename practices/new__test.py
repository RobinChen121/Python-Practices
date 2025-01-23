"""
Created on 2025/1/22, 18:25 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""


class ClassA:
    # def __new__(cls):
    #     print('ClassA.__new__')
    #     return super().__new__(cls)

    def __init__(self) -> None:
        print('ClassA.__init__')

    def __call__(self, *args):
        print('ClassA.__call__ args:', args)

class ClassB:
    def __new__(cls):
        print('ClassB.__new__')
        return super().__new__(ClassA)
        # return ClassA()  # 也可用这种写法

    def __init__(self):
        print('ClassB.__init__')


b = ClassB()
print(b)
print(type(b))

print('*' * 25)
cls = b.__class__
print(cls)
result = cls.__new__(cls)
print(result)
