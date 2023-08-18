#GRAFICA EL RESIDUAL PLOT PARA UNA REGRESIÓN POLINÓMICA MÚLTIPLE

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las variables predictoras (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df[['price']]

# Transformar las características en una matriz polinómica
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Ajustar el modelo de regresión lineal a los datos polinómicos
model = LinearRegression()
model.fit(x_poly, y)

# Calcular los valores predichos y los residuos
y_pred = model.predict(x_poly)
residuals = y - y_pred

# Crear el gráfico de residuos
plt.figure(figsize=(8, 6))
plt.scatter(y_pred, residuals, color='blue')
plt.axhline(y=0, color='red', linestyle='--')
plt.title('Residual Plot de la Regresión Polinómica Múltiple')
plt.xlabel('Valores Predichos')
plt.ylabel('Residuos')
plt.show()
