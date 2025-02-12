"""
Created on 2025/1/22, 11:22 

@author: Zhen Chen.

@Python version: 3.10

@disp:  

"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from sklearn.linear_model import LinearRegression

# Generate sample data
np.random.seed(42)
n_points = 50
x = np.linspace(0, 10, n_points)
y = 2 * x + 3 + np.random.normal(0, 2, n_points)  # y = 2x + 3 + noise

# Set up the figure and axis
fig, ax = plt.subplots()
ax.set_xlim(0, 10)
ax.set_ylim(0, 25)
ax.set_xlabel("X")
ax.set_ylabel("Y")
ax.set_title("Linear Regression Animation")

# Plot the data points
scatter = ax.scatter([], [], color='blue', label="Data Points")

# Plot the regression line
regression_line, = ax.plot([], [], 'r-', label="Regression Line")

# Add legend
ax.legend(loc = 'upper right')

# Text to display the current number of points
points_text = ax.text(0.02, 0.95, "", fontsize=12, transform=ax.transAxes, color = 'green')

# Function to update the animation
def animate(frame):
    # Use the first `frame` data points
    x_data = x[:frame]
    y_data = y[:frame]

    # Update the scatter plot
    scatter.set_offsets(np.column_stack((x_data, y_data)))

    # Fit linear regression if there are at least 2 points
    if frame >= 2:
        model = LinearRegression()
        model.fit(x_data.reshape(-1, 1), y_data)
        y_pred = model.predict(x.reshape(-1, 1))

        # Update the regression line
        regression_line.set_data(x, y_pred)

    # Update the alpha text
    points_text.set_text(f"points = {frame}")
    # points_text.set_text(f"Points: {frame}")

    return scatter, regression_line, points_text

# Create the animation
ani = FuncAnimation(fig, animate, frames=n_points + 1, interval=200, blit=True)

# Display the animation
plt.show()
ani.save('linear_regression.gif')