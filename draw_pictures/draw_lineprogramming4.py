"""
Created on 2025/2/9, 19:54 

@author: Zhen Chen.

@Python version: 3.10

@description:  

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Define x1 range
x1 = np.linspace(0, 10, 200)

# Fixed constraint
x2_1 = (8 - 2 * x1)  # 2x1 + x2 <= 8

# Prepare figure
fig, ax = plt.subplots(figsize=(8, 6))

# Plot the fixed constraint
ax.plot(x1, x2_1, label=r'$2x_1 + x_2 \leq 8$', color='blue')

# Placeholder for the dynamic constraint and feasible region
line, = ax.plot([], [], label=r'$x_1 + 3x_2 \leq b$', color='red')
feasible = ax.fill_between(x1, 0, 0, color='lightgrey', alpha=0.5, label='Feasible Region')

# Highlight the optimal point
point, = ax.plot([], [], 'o', color='purple', label='Optimal Point')

# Set up the plot
ax.set_xlim(0, 5)
ax.set_ylim(0, 5)
ax.set_title('Effect of Constraint Change on Feasible Region')
ax.set_xlabel(r'$x_1$')
ax.set_ylabel(r'$x_2$')
ax.axhline(0, color='black', linewidth=0.5)
ax.axvline(0, color='black', linewidth=0.5)
ax.legend()
ax.grid(True)

# Update function for the animation
def update(b):
    # Dynamic constraint: x1 + 3x2 <= b --> x2 = (b - x1) / 3
    x2_dynamic = (b - x1) / 3

    # Update the dynamic line
    line.set_data(x1, x2_dynamic)

    # Update the feasible region
    global feasible
    feasible.remove()  # Remove the old region
    feasible = ax.fill_between(
        x1,
        np.minimum(x2_1, x2_dynamic),
        0,
        where=(x2_1 >= 0) & (x2_dynamic >= 0),
        color='lightgrey',
        alpha=0.5
    )

    # Solve for the optimal point dynamically
    from scipy.optimize import linprog
    c = [-30, -20]
    A = [[2, 1], [1, 3]]
    b_values = [8, b]
    bounds = [(0, None), (0, None)]
    result = linprog(c, A_ub=A, b_ub=b_values, bounds=bounds, method='highs')

    # Update the optimal point
    if result.success:
        x_opt, y_opt = result.x
        point.set_data(x_opt, y_opt)

    return line, feasible, point

# Create the animation
anim = FuncAnimation(fig, update, frames=np.linspace(8, 9, 50), blit=False)

# Save as GIF or MP4 if needed
anim.save('constraint_change_animation.gif', fps=10)

plt.show()
