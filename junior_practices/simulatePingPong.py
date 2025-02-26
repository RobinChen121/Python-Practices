# -*- coding: utf-8 -*-
"""
Created on Thu Oct 14 17:51:16 2021

@author: zhen chen

MIT Licence.

Python version: 3.7


Description: 
    
"""
from random import *


def printIntro():  # 打印程序介绍信息
    print("10号张颖慧进行比赛分析结果：")
    print("这个程序模拟两个选手A和B的某种竞技比赛")
    print("程序运行需要A和B的能力值（以0到1之间的小数表示）")


def getInputs():  # 获得程序运行参数
    a = eval(input("请输入选手A的能力值(0-1): "))
    b = eval(input("请输入选手B的能力值(0-1): "))
    n = eval(input("模拟比赛的场次: "))
    return a, b, n


def simNGames(n, probA, probB):  # 进行N场比赛
    winsA, winsB = 0, 0
    for i in range(n):
        for j in range(7):  # 进行7局4胜的比赛
            scoreA, scoreB = simOneGame(probA, probB)
            if scoreA > scoreB:
                winsA += 1
            else:
                winsB += 1
    return winsA, winsB


def gameOver(a, b):  # 正常比赛结束
    return a == 11 or b == 11


def gameOver2(a, b):  # 进行抢12比赛结束
    if abs((a - b)) >= 2:
        return a, b


def simOneGame(probA, probB):  # 进行一场比赛
    scoreA, scoreB = 0, 0  # 初始化AB的得分
    serving = sample(["A", "B"], 1)
    num = 1
    while not gameOver(scoreA, scoreB):  # 用while循环来执行比赛
        if scoreA == 10 and scoreB == 10:
            return simtwoGame2(probA, probB, serving)
        if serving == "A":
            if random() < probA:  ##用随机数生成胜负
                scoreA += 1
            if num % 2 == 0:
                serving = "B"
        else:
            if random() < probB:
                scoreB += 1
            if num % 2 == 0:
                serving = "A"
        num = num + 1
    return scoreA, scoreB


def simtwoGame2(probA, probB, serving):
    scoreA, scoreB = 10, 10
    while not gameOver2(scoreA, scoreB):
        if serving == "A":
            if random() < probA:
                scoreA += 1
            serving = "B"
        else:
            if random() < probB:
                scoreB += 1
            serving = "A"
    return scoreA, scoreB


def printSummary(winsA, winsB):
    n = winsA + winsB
    print("竞技分析开始，共模拟{}场比赛".format(n))
    print("选手A获胜{}场比赛，占比{:0.1%}".format(winsA, winsA / n))
    print("选手B获胜{}场比赛，占比{:0.1%}".format(winsB, winsB / n))


def main():
    printIntro()
    probA, probB, n = getInputs()
    winsA, winsB = simNGames(n, probA, probB)
    printSummary(winsA, winsB)


main()
