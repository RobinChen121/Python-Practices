# -*- coding: utf-8 -*-
"""
Created on Tue May  7 17:51:05 2019

@author: zhen chen

Python version: 3.7

Description: use basemap
    
""" 

# must set the path for proj library
import os
os.environ['PROJ_LIB'] = r'C:\Users\chen_\AppData\Local\Continuum\anaconda3\Library\share'

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt


map = Basemap()
map.drawcoastlines()
#map.drawcounties(linewidth=1.5)

plt.show()
plt.savefig('test.png')

map = Basemap(projection='ortho', 
              lat_0=0, lon_0=0)

#Fill the globe with a blue color 
map.drawmapboundary(fill_color='aqua')
#Fill the continents with the land color
map.fillcontinents(color='coral',lake_color='aqua')

map.drawcoastlines()

plt.show()
