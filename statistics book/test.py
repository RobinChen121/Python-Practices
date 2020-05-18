""" 
# @File  : test.py
# @Author: Chen Zhen
# python version: 3.7
# @Date  : 2020/2/29
# @Desc  : test the code formulation of

"""

def changeSet(mySet):
    mySet.update([1, 2, 3])
    return mySet

setTry = {10, 20, 30}
changeSet(setTry)
print(setTry)  # 结果是 [10, 20, 30, 1, 2, 3]
