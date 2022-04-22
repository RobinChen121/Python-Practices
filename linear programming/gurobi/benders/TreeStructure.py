# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 10:36:42 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: 
    
"""
from anytree import Node, RenderTree


def get_tree_strcture(samples):
    T = len(samples[0])
    N = len(samples)
    node_values = [[] for t in range(T)]
    node_index = [[] for t in range(T)] # this is the wanted value
    for t in range(T):
        node_num = 0
        if t == 0:           
            for i in range(N):           
                if samples[i][t] not in node_values[t]:
                    node_values[t].append(samples[i][t]) 
                    node_index[t].append([])
                    node_index[t][node_num].append(i)
                    node_num = node_num + 1
                else:
                    temp_m = len(node_values[t])
                    for j in range(temp_m): # should revise
                        if samples[i][t] == node_values[t][j]:
                            node_index[t][j].append(i)
                            break
        else:
            lastNodeNum = len(node_index[t-1])
            for i in range(lastNodeNum):
                child_num = len(node_index[t-1][i])
                node_values[t].append([])
                for j in range(child_num):
                    index = node_index[t-1][i][j]
                    if samples[index][t] not in node_values[t][i]:
                        node_values[t][i].append(samples[index][t]) 
                        node_index[t].append([])
                        node_index[t][node_num].append(index)
                        node_num = node_num + 1
                    else:
                        temp_m = len(node_values[t][i]) #2
                        for k in range(temp_m): 
                            if samples[index][t] == node_values[t][i][k]:
                                node_index[t][k].append(index)
                                break
                    
    return node_values, node_index

def draw_tree(node_values):
    sta = Node("start")
    generation_num = len(node_values)
    last_nodes = [[] for t in range(generation_num)]
    for t in range(generation_num):            
        if t == 0:
            parent_num = 1
            child_num = len(node_values[t])
            for j in range(child_num):
                last_nodes[t].append(Node('t'+str(t+1), parent=sta, value = node_values[t][j]))
            
        else:
            parent_num = len(last_nodes[t-1]) 
            for i in range(parent_num):
                child_num = len(node_values[t][i])
                for j in range(child_num):
                    last_nodes[t].append(Node('t'+str(t+1), parent = last_nodes[t-1][i], value = node_values[t][i][j]))
    print(RenderTree(sta))
    


# samples = [[5, 2, 4], [5, 3, 4], [3, 3, 5], [5, 2, 8], [5, 3, 7]]
# node_values, node_index = get_tree_strcture(samples)
# draw_tree(node_values)