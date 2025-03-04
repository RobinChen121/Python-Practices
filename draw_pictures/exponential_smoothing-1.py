"""
Created on 2025/1/22, 10:45 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

matplotlib.use('TkAgg')  # 强制使用 Tkinter 窗口

# Parameters
n_points = 50  # Number of data points
time = np.linspace(0, 10, n_points)  # Time axis

# Generate sample data (e.g., a sine wave with noise)
np.random.seed(42)
data = np.sin(time) + np.random.normal(0, 0.2, n_points)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-2, 2)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title(r"Exponential Smoothing with Changing $\alpha$")

# Plot the original data
original_line, = ax.plot(time, data, 'o-', label="Original Data", alpha=0.5)  # 因为 ax.plot() 返回一个线列表，即使只有一条线

# Plot the smoothed data
smoothed_line, = ax.plot([], [], 'r-', label="Smoothed Data")

# Add legend
ax.legend()

# Text to display the current alpha value
alpha_text = ax.text(0.5, 1.5, "", fontsize=12)


# Function to compute smoothed data for a given alpha
def exponential_smoothing(data, alpha):
    smoothed_data = np.zeros_like(data)
    smoothed_data[0] = data[0]  # Initialize with the first data point
    for t in range(1, len(data)):
        smoothed_data[t] = alpha * data[t] + (1 - alpha) * smoothed_data[t - 1]
    return smoothed_data


# Animation function
def animate(alpha):
    # Compute smoothed data for the current alpha
    smoothed_data = exponential_smoothing(data, alpha)

    # Update the smoothed line
    smoothed_line.set_data(time, smoothed_data)

    # Update the alpha text
    alpha_text.set_text(r'$\alpha$ = ' + str(round(alpha, 2)))

    return smoothed_line, alpha_text


# Create the animation
alphas = np.linspace(0.1, 0.99, 100)  # Range of alpha values to animate
ani = FuncAnimation(fig, animate, frames=alphas, interval=100, blit=True, repeat = False)
# ani.save('smoothing-1.gif')

# Display the animation
plt.show()