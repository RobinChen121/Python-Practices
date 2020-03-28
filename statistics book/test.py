""" 
# @File  : test.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/2/29
# @Desc  : test the code formulation of

"""

wage = int(input("请输入你的月薪: "))
print("")
if wage <= 3000:
    print("贫困户")
    if 2000 <= wage < 3000:
        print("贫困户中的一般贫")
    if wage <= 1000:
        print("贫困户中的特贫")
else:
    print("不是贫困户")