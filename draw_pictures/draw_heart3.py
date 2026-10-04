#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Nov  8 11:35:14 2023

@author: zhen chen

@disp:


"""

# type %matplotlib qt in Jupyter to shown figure in a separate window
# To use %matplotlib qt, make sure you have the required PyQt library installed in your environment (pip install PyQt5 or PyQt6)
# if not using Juper notebook, put two addition lines of codes:
# import matplotlib
# matplotlib.use('QtAgg')

from matplotlib.animation import FuncAnimation
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# you may need the following two lines of code to show the figure in a separate window
import matplotlib
matplotlib.use('QtAgg')

# Apply the default theme
sns.set_theme("paper")
sns.set_context("paper")
frames = 50 # number of frames in the animation，meaning the number of animation function calls

def animate(alpha):
    x = np.linspace(-1.8, 1.8, 1000)
    y = abs(x) ** (2 / 3) + 0.9 * np.sqrt(3.3 - x**2) * np.sin(alpha * (np.pi) * x)
    PLOT.set_data(x, y)
    # dynamic color change logic: as alpha increases, the color changes from red to purple
    blue_val = min(1, alpha / frames + 0.2)
    red_val = max(0, 1 - blue_val)
    PLOT.set_color((blue_val, 0.1, red_val))  # change color
    time_text.set_text(r"$\alpha$ = " + str(round(alpha, 2)))
    return PLOT, time_text


fig = plt.figure()
ax = fig.add_subplot(111, xlim=(-2.5, 2.5), ylim=(-2, 4))  # or plt.subplot
(PLOT,) = ax.plot([], [])  # return all the lines
plt.text(-1, 3, r"$f(x)=x^{2/3}+0.9(3.3-x^2)^{1/2}\sin(\alpha\pi x)$")
time_text = ax.text(-0.25, 2.5, "")  # transform = ax.transAxes

ani = FuncAnimation(fig, animate, frames=frames, interval=200, repeat=True)
plt.show()
ani.save("heart.gif")  # save fig as a gif file
