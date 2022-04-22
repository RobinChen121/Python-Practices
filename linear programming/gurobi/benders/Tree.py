# -*- coding: utf-8 -*-
"""
Created on Wed Feb 23 19:15:30 2022

@author: zhen chen
@Email: chen.zhen5526@gmail.com

MIT Licence.

Python version: 3.8


Description: the codes are imperfect
    
"""


class Node():
    def __init__(self, data):
        self.data = data
        self.children = []
        self.ancestor = []
        self.t_index = 0 # which period of this node
        self.i_index = 0 # the node index for this node in its period   
        self.scenrio_num = None # number of scenarios from this node as root
        self.generation_num = None # maximum generation number from this node as root
        
    def __str__(self): # for printing
        solid_line = "".join([chr(0x2015) for c in range(2)])
        string = 'root\n' +'|'
        this_node = self
        while len(this_node.children) > 0:
            string = string + solid_line + ' '+ str(this_node.children[0].data)
            this_node = this_node.children[0]
        string = string + '\n'
        return string
                
    def setChildren(self, datas):
        child_num = len(datas)
        for i in range(child_num):
            self.children.append(Node(datas[i]))   
            
    def addPath(self, datas):
        this_node = Node(datas[0])
        self.children.append(this_node)
        i = 1
        while i < len(datas):            
            this_node.children.append(Node(datas[i]))
            this_node = this_node.children[0]
            i = i + 1
        self.generation_num = i
            
a = Node('root')
a.addPath([5, 2, 3])   
print(a) 
    