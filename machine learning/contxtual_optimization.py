"""
@Python version: 3.13
@Author: Zhen Chen
@Email: chen.zhen5526@gmail.com
@Time: 20/10/2026, 21:20
@Desc:

"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1000

# Weather:
# 0 = Rainy
# 1 = Cloudy
# 2 = Sunny

weather = np.random.choice([0, 1, 2], size=N)

# Conditional Poisson parameters
lambda_values = np.array([10, 20, 30])

# Generate demand
demand = np.random.poisson(lambda_values[weather])

data = pd.DataFrame({"weather": weather, "demand": demand})

print(data.head())

from sklearn.linear_model import PoissonRegressor

# One-hot encoding for categorical weather
# 将类别变量转换为 one-hot 编码，即将每个类别变量转换为一个二进制向量，其中只有一个元素为1，其余元素为0。
X = pd.get_dummies(data["weather"], prefix="weather").astype(float)

y = data["demand"]

model = PoissonRegressor(alpha=0.001, max_iter=1000)

model.fit(X, y)

# Predict conditional demand means
predicted_lambda = model.predict(X)

print(predicted_lambda[:10])


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import poisson
from sklearn.linear_model import PoissonRegressor
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_poisson_deviance

# =========================================================
# 1. Generate simulated demand data
# =========================================================

rng = np.random.default_rng(42)

n = 500

# Temperature between 5 and 30 degrees
temperature = rng.uniform(5, 30, n)

# True Poisson regression parameters
beta0 = 1.0
beta1 = 0.05

# Conditional mean of demand
lambda_true = np.exp(beta0 + beta1 * temperature)

# Generate random demand
demand = rng.poisson(lambda_true)

data = pd.DataFrame({"temperature": temperature, "demand": demand})

print("First five observations:")
print(data.head())


# =========================================================
# 2. Train Poisson regression
# =========================================================

X = data[["temperature"]]
y = data["demand"]

model = PoissonRegressor(alpha=0, max_iter=1000)

model.fit(X, y)

print("\nEstimated parameters:")
print("True intercept:", beta0)
print("Estimated intercept:", model.intercept_)

print("True coefficient:", beta1)
print("Estimated coefficient:", model.coef_[0])


# =========================================================
# 3. Predict conditional mean
# =========================================================

new_data = pd.DataFrame({"temperature": [20]})

lambda_pred = model.predict(new_data)[0]  # 返回一个一元素数组，取第一个元素作为预测值

print("\nPrediction at 20°C:")
print("Predicted mean demand:", lambda_pred)
print("True mean demand:", np.exp(beta0 + beta1 * 20))


# =========================================================
# 4. Visualize Poisson regression
# =========================================================

temperature_grid = np.linspace(5, 30, 100)

X_grid = pd.DataFrame({"temperature": temperature_grid})

lambda_pred_grid = model.predict(X_grid)

plt.figure(figsize=(10, 6))

plt.scatter(temperature, demand, alpha=0.3, label="Observed demand")

plt.plot(
    temperature_grid,
    lambda_pred_grid,
    color="red",
    linewidth=2,
    label="Poisson regression",
)

plt.plot(
    temperature_grid,
    np.exp(beta0 + beta1 * temperature_grid),
    color="green",
    linestyle="--",
    label="True conditional mean",
)

plt.xlabel("Temperature")
plt.ylabel("Demand")
plt.title("Poisson Regression")
plt.legend()
plt.grid(alpha=0.3)
plt.show()


# =========================================================
# 5. Calculate demand probabilities
# =========================================================

print("\nDemand distribution at 20°C:")

# Probability that demand equals 10
prob_equal_10 = poisson.pmf(10, mu=lambda_pred)

# Probability that demand is at most 10
prob_at_most_10 = poisson.cdf(10, mu=lambda_pred)

print("P(D = 10):", prob_equal_10)
print("P(D <= 10):", prob_at_most_10)


# =========================================================
# 6. Generate demand scenarios
# =========================================================

scenarios = poisson.rvs(mu=lambda_pred, size=100, random_state=42)

print("\nGenerated demand scenarios:")
print(scenarios[:20])

plt.figure(figsize=(9, 5))

plt.hist(
    scenarios,
    bins=np.arange(scenarios.min() - 0.5, scenarios.max() + 1.5, 1),
    edgecolor="black",
)

plt.xlabel("Demand")
plt.ylabel("Frequency")
plt.title("Generated Demand Scenarios")
plt.show()


# =========================================================
# 7. Poisson regression with categorical variables
# =========================================================

weather_data = pd.DataFrame(
    {
        "weather": ["Sunny", "Rainy", "Cloudy", "Sunny", "Rainy"],
        "demand": [30, 10, 20, 35, 8],
    }
)

X_weather = pd.get_dummies(
    weather_data["weather"], prefix="weather", drop_first=True, dtype=int
)

print("\nOne-hot encoded weather data:")
print(X_weather)

weather_model = PoissonRegressor(alpha=0.01, max_iter=1000)

weather_model.fit(X_weather, weather_data["demand"])

print("\nWeather model intercept:")
print(weather_model.intercept_)

print("Weather model coefficients:")
print(weather_model.coef_)

# Predict demand for each weather category
weather_categories = ["Cloudy", "Rainy", "Sunny"]

new_weather = pd.DataFrame({"weather": weather_categories})

X_new_weather = pd.get_dummies(new_weather["weather"], prefix="weather", dtype=int)

X_new_weather = X_new_weather.reindex(columns=X_weather.columns, fill_value=0)

weather_predictions = weather_model.predict(X_new_weather)

print("\nPredicted demand by weather:")

for weather, prediction in zip(weather_categories, weather_predictions):
    print(f"{weather}: {prediction:.2f}")


# =========================================================
# 8. Select alpha using cross-validation
# =========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

param_grid = {"alpha": [0, 0.001, 0.01, 0.1, 1, 10]}

grid = GridSearchCV(
    estimator=PoissonRegressor(max_iter=1000),
    param_grid=param_grid,
    scoring="neg_mean_poisson_deviance",
    cv=5,
)

grid.fit(X_train, y_train)

print("\nBest alpha:", grid.best_params_["alpha"])

best_model = grid.best_estimator_

y_pred = best_model.predict(X_test)

test_deviance = mean_poisson_deviance(y_test, y_pred)

print("Test mean Poisson deviance:", test_deviance)

# Show CV results
cv_results = pd.DataFrame(grid.cv_results_)

print("\nCross-validation results:")
print(cv_results[["param_alpha", "mean_test_score", "rank_test_score"]])
