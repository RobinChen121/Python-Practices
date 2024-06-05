# -*- coding: utf-8 -*-
"""
Created on Thu May  2 22:51:51 2024

@author: chen
"""
import subprocess

import subprocess
def runcmd(command):
    ret = subprocess.run(command,shell=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,encoding="utf-8",timeout=1)
    if ret.returncode == 0:
        print("success:",ret)
    else:
        print("error:",ret)


# runcmd(["dir","/b"])#序列参数
runcmd("exit 1")#字符串参数