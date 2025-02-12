"""
created on 2025/2/3, 14:54
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define x values
x1 = np.linspace(0, 100, 200)

# Define the constraint lines
x2_1 = (240 - 3*x1) / 8  # From 3x1 + 8x2 = 240
x2_2 = np.full_like(x1, 15)  # x2 >= 15 (horizontal line at x2 = 15)

# Define the figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(0, 100)
ax.set_ylim(0, 50)

# Fill the feasible region
x1_vals = [0, 40, 0]  # x-coordinates of feasible region
x2_vals = [15, 15, 30]  # y-coordinates of feasible region
ax.fill(x1_vals, x2_vals, 'lightblue', alpha=0.5, label='Feasible Region')

# Plot the static lines
ax.plot(x1, x2_1, 'r', label='3x1 + 8x2 = 240')
ax.plot(x1, x2_2, 'g', label='x2 = 15')

# Create an animated line for 5x1 + 10x2 = k
line, = ax.plot([], [], 'b--', label='5x1 + 10x2 = k')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    k = frame
    x2_3 = (k - 5*x1) / 10
    line.set_data(x1, x2_3)
    return line,

# Animate the line moving from k=0 to k=150
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 350, 50), init_func=init, repeat = False)

# Labels and legend
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_title("Feasible Region with Animated Line")
ax.grid(True, linestyle='--', alpha=0.6)
# ani.save('line.gif')

plt.show()