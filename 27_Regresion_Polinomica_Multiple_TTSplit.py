# DIVIDE EL DATASET EN TRAIN Y TEST SETS, HACE PREDICCIONES USANDO REGRESIÓN POLINÓMICA MÚLTIPLE, CALCULA EL MSE Y R2 Y GRAFICA EL RESULTADO.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df[['price']]

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Transformar las características en una matriz polinómica
poly = PolynomialFeatures(degree=2)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Ajustar el modelo de regresión lineal a los datos polinómicos
model = LinearRegression()
model.fit(x_train_poly, y_train)

# Predecir valores utilizando el modelo ajustado
y_pred_train = model.predict(x_train_poly)
y_pred_test = model.predict(x_test_poly)

# Calcular el MSE en los conjuntos de entrenamiento y prueba
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)

# Calcular el R Squared en los conjuntos de entrenamiento y prueba
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print("R Squared (Train):", r2_train)
print("R Squared (Test):", r2_test)

# Gráfico
# Crear una figura con varios subgráficos
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

# Variables independientes para la visualización
independent_vars = ['horsepower', 'curb-weight', 'engine-size', 'width']

# Hacer subgráficos para cada variable independiente
for i, var in enumerate(independent_vars):
    # Scatter plot de los datos de entrenamiento y prueba
    axes[i].scatter(x_train[var], y_train, color='blue', label='Training Data')
    axes[i].scatter(x_test[var], y_test, color='green', label='Test Data')

    # Ajustar la regresión polinómica en los datos de entrenamiento
    poly_features = poly.fit_transform(x_train[[var]])
    model.fit(poly_features, y_train)
    y_pred_train_var = model.predict(poly_features)

    # Trazar la curva de regresión polinómica en los gráficos de entrenamiento y prueba
    x_range = np.linspace(min(x_train[var]), max(x_train[var]), num=100)
    x_range_poly = poly.transform(x_range.reshape(-1, 1))
    y_range_pred = model.predict(x_range_poly)

    axes[i].plot(x_range, y_range_pred, color='orange', label='Regresión polinómica')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Price')
    axes[i].legend()

plt.tight_layout()
plt.show()
