"""
Created on 2025/1/22, 11:11 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
n_points = 50  # Number of data points
time = np.linspace(0, 10, n_points)  # Time axis

# Generate sample data (e.g., a sine wave with noise and a trend)
np.random.seed(42)
data = np.sin(time) + 0.1 * time + np.random.normal(0, 0.2, n_points)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-2, 3)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title(r"Double Exponential Smoothing: Effect of $\beta$)")

# Plot the original data
original_line, = ax.plot(time, data, 'o-', label="Original Data", alpha=0.5)

# Plot the forecasted data
forecast_line, = ax.plot([], [], 'g--', label="Forecast")

# Add legend
ax.legend(loc = 'upper right')

# Text to display the current beta value
beta_text = ax.text(0.02, 0.9, "", fontsize=12, transform=ax.transAxes, color='purple')

# Function to compute double exponential smoothing
def double_exponential_smoothing(data, alpha, beta):
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    forecast = np.zeros(n)

    # Initialize level and trend
    level[0] = data[0]
    trend[0] = data[1] - data[0]

    # Apply double exponential smoothing
    for t in range(1, n):
        level[t] = alpha * data[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        forecast[t] = level[t - 1] + trend[t - 1]

    return forecast

# Animation function
def animate(beta):
    alpha = 0.2 # Fixed alpha for simplicity

    # Compute forecasted data
    forecast = double_exponential_smoothing(data, alpha, beta)

    # Update the forecast line
    forecast_line.set_data(time, forecast)

    # Update the beta text
    beta_text.set_text(r"$\alpha ={}$, $\beta$ = {:.2f}".format(alpha, beta))

    return forecast_line, beta_text

# Create the animation
betas = np.linspace(0.1, 0.99, 100)  # Range of beta values to animate
ani = FuncAnimation(fig, animate, frames=betas, interval=100, blit=True)

# Display the animation
plt.show()
ani.save('2-smoothing-1.gif')


# second
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Parameters
n_points = 50  # Number of data points
time = np.linspace(0, 10, n_points)  # Time axis

# Generate sample data (e.g., a sine wave with noise and a trend)
np.random.seed(42)
data = np.sin(time) + 0.1 * time + np.random.normal(0, 0.2, n_points)

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(-2, 3)
ax.set_xlabel("Time")
ax.set_ylabel("Value")
ax.set_title(r"Double Exponential Smoothing: Effect of $\alpha$")

# Plot the original data
original_line, = ax.plot(time, data, 'o-', label="Original Data", alpha=0.5)

# Plot the smoothed data
# smoothed_line, = ax.plot([], [], 'r-', label="Smoothed Data")

# Plot the forecasted data
forecast_line, = ax.plot([], [], 'g--', label="Forecast")

# Add legend
ax.legend(loc='upper right')

# Text to display the current alpha and beta values
beta_text = ax.text(0.02, 0.9, "", fontsize=12, transform=ax.transAxes, color='purple')

# Function to compute double exponential smoothing
def double_exponential_smoothing(data, alpha, beta):
    n = len(data)
    level = np.zeros(n)
    trend = np.zeros(n)
    forecast = np.zeros(n)

    # Initialize level and trend
    level[0] = data[0]
    trend[0] = data[1] - data[0]

    # Apply double exponential smoothing
    for t in range(1, n):
        level[t] = alpha * data[t] + (1 - alpha) * (level[t - 1] + trend[t - 1])
        trend[t] = beta * (level[t] - level[t - 1]) + (1 - beta) * trend[t - 1]
        forecast[t] = level[t - 1] + trend[t - 1]

    return level, trend, forecast

# Animation function
def animate(frame):
    alpha = frame / 100  # Vary alpha from 0 to 1
    beta = 0.2  # Fixed beta for simplicity (can also be varied)

    # Compute smoothed data and forecast
    level, trend, forecast = double_exponential_smoothing(data, alpha, beta)

    # Update the smoothed line
    # smoothed_line.set_data(time, level)

    # Update the forecast line (forecast one step ahead)
    forecast_line.set_data(time, forecast)

    # Update the alpha and beta text
    beta_text.set_text(r"$\alpha ={:.2f}$, $\beta$ = {}".format(alpha, beta))


    return forecast_line, beta_text

# Create the animation
ani = FuncAnimation(fig, animate, frames=100, interval=100, blit=True)

# Display the animation
plt.show()
# ani.save('2-smoothing-2.gif')