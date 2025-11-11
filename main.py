import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('data.csv')

data['km_scaled'] = (data['km'] - data['km'].mean()) / data['km'].std()
data['price_scaled'] = (data['price'] - data['price'].mean()) / data['price'].std()

def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0

    n = len(points)

    for i in range(n):
        x = points.iloc[i].km_scaled
        y = points.iloc[i].price_scaled

        m_gradient += -(2/n) * x * (y - (m_now * x + b_now)) #partial derivative of Error function with respect to m
        b_gradient += -(2/n) * (y-(m_now * x + b_now)) #partial derivative of Error function with respect to b
    #finding the opposite of the partial derivative / maximum value
    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

m = 0
b = 0
L = 0.01 
epochs = 1000

for i in range(epochs):
    m, b = gradient_descent(m, b, data, L)
    if i % 100 == 0:
        print(f"Epoch: {i}")
print(m, b)

predicted_scaled = m * data['km_scaled'] + b
predicted = predicted_scaled * data['price'].std() + data['price'].mean()

plt.scatter(data.km, data.price, color="black")
plt.plot(data.km, predicted, color='red')
plt.xlabel("Kilometers")
plt.ylabel("Price (€)")
plt.show()