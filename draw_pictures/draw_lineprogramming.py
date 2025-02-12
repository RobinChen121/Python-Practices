"""
created on 2025/2/3, 14:24
@author: Zhen Chen, chen.zhen5526@gmail.com
@version: 3.10

@description:

"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# Define x values
x1 = np.linspace(-2, 5, 200)

# Define the constraint lines
x2_1 = (4 - 2*x1)  # From 2x1 + x2 = 4
x2_2 = (5 - x1) / 2  # From x1 + 2x2 = 5

# Define the figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(-2, 5)
ax.set_ylim(-2, 5)

# Fill the feasible region
x1_vals = [0, 2, 1, 0]  # x-coordinates of feasible region
x2_vals = [0, 0, 2, 2.5]  # y-coordinates of feasible region
ax.fill(x1_vals, x2_vals, 'lightblue', alpha=0.5, label='Feasible Region')

# Plot the static lines
ax.plot(x1, x2_1, 'r', label='2x1 + x2 = 4')
ax.plot(x1, x2_2, 'g', label='x1 + 2x2 = 5')

# Create an animated line for 2x1 + 3x2 = k
line, = ax.plot([], [], 'b--', label='2x1 + 3x2 = k')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    k = frame
    x2_3 = (k - 2*x1) / 3
    line.set_data(x1, x2_3)
    return line,

# Animate the line moving from k=0 to k=8
ani = animation.FuncAnimation(fig, update, frames=np.linspace(0, 8, 50), init_func=init, blit=True)

# Mark intersection points
points = [(0, 4), (2, 0), (0, 2.5), (1, 2)]
for (x, y) in points:
    ax.scatter(x, y, color='black', zorder=3)
    ax.text(x + 0.1, y, f'({x},{y})', fontsize=10)

# Labels and legend
ax.set_xlabel("x1")
ax.set_ylabel("x2")
ax.axhline(0, color='black', linewidth=1)
ax.axvline(0, color='black', linewidth=1)
ax.legend()
ax.set_title("Feasible Region for Given Constraints")
ax.grid(True, linestyle='--', alpha=0.6)
# ani.save('line.gif')

plt.show()
