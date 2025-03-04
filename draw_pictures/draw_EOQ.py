"""
Python version: 3.12.7
Author: Zhen Chen, chen.zhen5526@gmail.com
Date: 2025/2/27 17:52
Description: 
    

"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg")


K = 100
D = 100
h = 200
c = 0


def costs(q):
    order_costs_ = K * D / q + c * D
    hold_costs_ = h * q / 2
    total_costs = order_costs_ + hold_costs_
    return order_costs_, hold_costs_, total_costs


Q = np.arange(1, 50, 1)
order_costs, hold_costs, total = costs(Q)

plt.plot(Q, order_costs, 'b', label="oder costs")
plt.plot(Q, hold_costs, 'g',label="hold costs")
plt.plot(Q, total, 'r', label="total costs")
plt.scatter(10, 2000)
plt.scatter(10, 1000)
plt.legend()
plt.show()

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 50)
ax.set_ylim(0, 10000)
ax.set_xlabel("Ordering quantity")
ax.set_ylabel("Costs")
ax.set_title(r"Costs related with Q")

order_costs_line, = ax.plot([], [], 'b', label="oder costs")
hold_costs_line, = ax.plot([], [], 'g', label="hold costs")
total_costs_line, = ax.plot([], [], 'r', label="total costs")

# Animation function
def animate(Q_):
    order_costs_, hold_costs_, total_ = costs(np.arange(1, Q_ + 1))

    # Update the lines
    order_costs_line.set_data(np.arange(1, Q_ + 1), order_costs_)
    hold_costs_line.set_data(np.arange(1, Q_ + 1), hold_costs_)
    total_costs_line.set_data(np.arange(1, Q_ + 1), total_)

    return order_costs_line, hold_costs_line, total_costs_line


# Create the animation
Q = np.arange(1, 50, 1)
ani = FuncAnimation(fig, animate, frames=Q)
ani.save('EOQ.gif')

# Display the animation
plt.show()
