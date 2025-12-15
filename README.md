# Car_Price_Prediction
An implementation of linear regression for car price prediction

## Linear Regression with Gradient Descent (Car Price Estimator)

This project implements a **simple linear regression model** from scratch to estimate car prices based on mileage using **gradient descent**.

### Project Components
- `train.py` — trains the model and saves it
- `predict.py` — loads the trained model and predicts a price
- No machine-learning libraries used for training — only basic math

### Project Structure
```
.
├── data.csv           # Training data (km, price)
├── train.py           # Model training
├── predict.py         # Price prediction
├── model.json         # Saved trained model
└── README.md
```

---

## The Model

**Linear regression formula:**
```
ŷ = θ₀ + θ₁x
```

| Variable | Definition |
|----------|-----------|
| x | Car mileage (km) |
| ŷ | Predicted price |
| θ₀ | Intercept |
| θ₁ | Slope |

**Goal:** Find θ₀ and θ₁ values that minimize prediction error.

---

## Feature Scaling

**Standardization formula:**
```
x_scaled = (x − μ) / σ
```

**Benefits:**
- Prevents large numbers from slowing learning
- Makes gradient descent converge faster
- Avoids numerical instability

> The model trains on scaled data, but predictions convert back to real prices.

---

## Gradient Descent

**Error function (Mean Squared Error):**
```
J(θ) = (1 / n) Σ (ŷ − y)²
```

**Partial derivatives:**
- `∂J/∂θ₀ = (1 / n) Σ (ŷ − y)`
- `∂J/∂θ₁ = (1 / n) Σ (ŷ − y)x`

**Update rule:**
```
θ := θ − L · ∇J
```

Where L = learning rate, ∇J = gradient

---

## Hyperparameters

| Parameter | Value | Purpose |
|-----------|-------|---------|
| Learning Rate (L) | 0.1 | Controls step size |
| Epochs | 1000 | Passes over dataset |

---

## Training Process (train.py)

1. Load and validate `data.csv`
2. Scale km and price
3. Initialize parameters (theta0, theta1)
4. Run gradient descent for N epochs
5. Save model parameters and scaling values to `model.json`

---

## Prediction (predict.py)

**Steps:**
1. Load `model.json`
2. Scale input mileage
3. Apply linear equation
4. Convert price back to real scale
5. Display estimated price

**Prediction formula:**
```
price = (θ₀ + θ₁ · km_scaled) · price_std + price_mean
```
