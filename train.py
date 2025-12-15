import pandas as pd
import json
import sys

DATA_FILE = "data.csv"
MODEL_FILE = "model.json"

def load_data(path):
    try:
        data = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: data.csv not found.")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        print("Error: data.csv is empty.")
        sys.exit(1)

    if data.empty:
        print("Error: data.csv has no rows.")
        sys.exit(1)

    required_cols = {"km", "price"}
    if not required_cols.issubset(data.columns):
        print("Error: CSV must contain 'km' and 'price' columns.")
        sys.exit(1)

    try:
        data = data.astype(float)
    except ValueError:
        print("Error: 'km' and 'price' must be numeric.")
        sys.exit(1)

    return data


def gradient_descent(theta0, theta1, points, L):
    n = len(points)

    if n == 0:
        return theta0, theta1

    error = theta0 + theta1 * points["km"] - points["price"]
    theta0 -= L * (1 / n) * error.sum()
    theta1 -= L * (1 / n) * (error * points["km"]).sum()

    return theta0, theta1


def main():
    data = load_data(DATA_FILE)

    km_mean = data["km"].mean()
    km_std = data["km"].std()
    price_mean = data["price"].mean()
    price_std = data["price"].std()

    if km_std == 0 or price_std == 0:
        print("Error: Standard deviation is zero (invalid data).")
        sys.exit(1)

    data["km"] = (data["km"] - km_mean) / km_std
    data["price"] = (data["price"] - price_mean) / price_std

    theta0 = 0.0
    theta1 = 0.0
    L = 0.1
    epochs = 1000

    for _ in range(epochs):
        theta0, theta1 = gradient_descent(theta0, theta1, data, L)

    model = {
        "theta0": theta0,
        "theta1": theta1,
        "km_mean": km_mean,
        "km_std": km_std,
        "price_mean": price_mean,
        "price_std": price_std,
    }

    with open(MODEL_FILE, "w") as f:
        json.dump(model, f, indent=4)

    print("Training completed.")
    print(f"Saved model to {MODEL_FILE}")
    print(f"theta0 = {theta0}, theta1 = {theta1}")


if __name__ == "__main__":
    main()
