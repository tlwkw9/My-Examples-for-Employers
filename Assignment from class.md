# First thing first ALWAYS ADD THE NECCESSARY LIBRARIES.

import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor
import pandas as pd

# Step 1. Generate data

np.random.seed(0)

x = np.random.uniform(1, 11, 100)
y = np.log(x) + np.random.normal(0, 0.2, 100)

x_values = np.array([1, 3, 5, 7, 9])   # prediction points
x_train = x.reshape(-1, 1)
x_test = x_values.reshape(-1, 1)


# Step 2. Define schemes

schemes = [
    "Equal contribution for K neighbors",
    "Inverse distance weighing for K neighbors",
    "All N points contribute with weighted contributions"
]

xy_pairs = []   # store (scheme, K, list of (x, yhat))


# Step 3.  Applying Gaussian kernel function

def gaussian_kernel_predict(x_train, y_train, x0, h=1.0):
    distances = x_train.flatten() - x0
    weights = np.exp(-(distances**2) / (2 * h**2))
    return np.sum(weights * y_train) / np.sum(weights)

# Step 4. Computing predictions for all 7 example cases

for scheme in schemes:

    if scheme == "Equal contribution for K neighbors":
        for K in [1, 3, 50]:
            model = KNeighborsRegressor(n_neighbors=K, weights='uniform')
            model.fit(x_train, y)
            y_pred = model.predict(x_test)
            xy_pairs.append((scheme, K, list(zip(x_values, y_pred))))

    elif scheme == "Inverse distance weighing for K neighbors":
        for K in [1, 3, 50]:
            model = KNeighborsRegressor(n_neighbors=K, weights='distance')
            model.fit(x_train, y)
            y_pred = model.predict(x_test)
            xy_pairs.append((scheme, K, list(zip(x_values, y_pred))))

    elif scheme == "All N points contribute with weighted contributions":
        y_pred = [gaussian_kernel_predict(x_train, y, x0) for x0 in x_values]
        xy_pairs.append((scheme, "Gaussian", list(zip(x_values, y_pred))))

# Step 5. Print at least 35 (x, y hat) pairs clean


rows = []
for scheme, K, pairs in xy_pairs:
    for x0, yhat in pairs:
        rows.append([scheme, K, x0, yhat])

df_results = pd.DataFrame(rows, columns=["Scheme", "K", "x", "y_hat"])
print(df_results)


# Step 6.  Function to get nearest (x0, y0) points

def get_x0_y0_for_targets(x_train, y_train, x_targets):
    x0_list = []
    y0_list = []
    for xt in x_targets:
        idx = np.argmin(np.abs(x_train.flatten() - xt))
        x0_list.append(x_train.flatten()[idx])
        y0_list.append(y_train[idx])
    return np.array(x0_list), np.array(y0_list)

# Precompute nearest neighbors once
x0s, y0s = get_x0_y0_for_targets(x_train, y, x_values)

# Step 7. Here I generated all 7 plots 

for scheme, K, pairs in xy_pairs:
    plt.figure(figsize=(5, 4))

    # predicted points (x, y_hat)
    xs, ys = zip(*pairs)
    plt.scatter(xs, ys, color='red', s=60, label="Predicted (x, ŷ)")

    # nearest sample points (x0, y0)
    plt.scatter(x0s, y0s, color='blue', marker='x', s=80, label="Nearest sample (x₀, y₀)")

    plt.title(f"{scheme}, K={K}")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.legend()
    plt.grid(True)
    plt.show()

