#REALIZA PREDICCIONES USANDO VALIDACIÓN CRUZADA PARA UNA REGRESIÓN POLINÓMICA

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower']]
y = df['price']

# Crear un modelo de regresión polinómica
degree = 2
poly = PolynomialFeatures(degree)
x_poly = poly.fit_transform(x)
model = LinearRegression()

# Obtener predicciones utilizando cross_val_predict()
predicted_prices = cross_val_predict(model, x_poly, y, cv=5)
print("Predicciones:", predicted_prices)

# Graficar los resultados
plt.scatter(x, y, color='blue', label='Datos reales')
plt.scatter(x, predicted_prices, color='red', label='Predicciones')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.legend()
plt.show()
