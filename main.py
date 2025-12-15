import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

#saving mean and std for normalization
km_mean = data['km'].mean()
km_std = data['km'].std()
price_mean = data['price'].mean()
price_std = data['price'].std()


#x - xmin / (xmax - xmin)

data['km'] = (data['km'] - km_mean) / km_std
# data['km'] = data['km'] - data['km'].min() / (data['km'].max() - data['km'].min())
# data['price']
data['price'] = (data['price'] - price_mean) / price_std

def gradient_descent(theta0, theta1, points, L):
    n = len(points)

    tmp_theta0 = L * (1/n) * sum(theta0 + theta1 * points.iloc[i].km - points.iloc[i].price for i in range(n))
    tmp_theta1 = L * (1/n) * sum((theta0 + theta1 * points.iloc[i].km - points.iloc[i].price) * points.iloc[i].km for i in range(n))
    #finding the opposite of the partial derivative / maximum value
    theta0 -= tmp_theta0
    theta1 -= tmp_theta1
    return theta0, theta1

def predict_price(mileage):
    scaled_mileage = (mileage - km_mean) / km_std
    price_scaled = theta0 + theta1 * scaled_mileage
    price = price_scaled * price_std + price_mean
    return price

theta0 = 0 #intercept
theta1 = 0 #slope
L = 0.1 
epochs = 1000

for i in range(epochs):
    theta0, theta1 = gradient_descent(theta0, theta1, data, L)

print(theta0, theta1)

predicted = theta0 + data['km'] * theta1

input_mileage = float(input("Enter car mileage (in km): "))
estimated_price = predict_price(input_mileage)
print(f"Estimated price for a car with {input_mileage} km: €{estimated_price:.2f}")

plt.scatter(data.km, data.price, color="black")
plt.plot(data.km, predicted, color='red')
plt.xlabel("Kilometers")
plt.ylabel("Price (€)")
plt.show()