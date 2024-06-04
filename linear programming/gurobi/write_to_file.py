#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  3 16:17:33 2024

@author: zhenchen

@disp:  
    
    
"""

import csv


def write_to_csv(file_address_name:str, header:list[str], content_array:list, first_write:bool):
    # check if content_array is 2D list or 1D list
    twoD = False
    if isinstance(content_array[0], list):
        twoD = True
    with open (file_address_name, 'a', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)   
        if first_write == True:
            csvwriter.writerow(header)
        if twoD:
            csvwriter.writerows(content_array)
        else:
            csvwriter.writerow(content_array)    
    return