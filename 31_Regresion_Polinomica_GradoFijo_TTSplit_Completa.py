# DIVIDE EL DATASET EN TRAIN Y TEST SETS, HACE PREDICCIONES USANDO REGRESIÓN POLINÓMICA, CALCULA EL MSE Y R2 Y GRAFICA EL RESULTADO JUNTO CON EL RESIDUAL PLOT Y EL DISTRIBUTION PLOT.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower']]
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

# Calcular los residuales
residuos_train = y_train.values - y_pred_train
residuos_test = y_test.values - y_pred_test

# Gráfico 1
# Valores para graficar la curva polinómica
x_range = np.linspace(df['horsepower'].min(), df['horsepower'].max(), num=100)
x_range_poly = poly.transform(x_range.reshape(-1, 1))
y_range_pred = model.predict(x_range_poly)
# Dispersión
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='green', label='Test Data')
plt.plot(x_range, y_range_pred, color='orange', label='Regresión polinómica')
plt.xlabel('Horsepower')
plt.ylabel('Price')
plt.legend()

# Gráfico 2
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Distribution Plot
sns.distplot(y, hist=False, color='red', label='Datos reales', ax=ax1)
sns.distplot(y_pred_train, hist=False, color='blue', label='Predicciones de Entrenamiento', ax=ax1)
sns.distplot(y_pred_test, hist=False, color='green', label='Predicciones de Prueba', ax=ax1)
ax1.set_xlabel('Precio')
ax1.set_ylabel('Densidad')
ax1.legend()

# Residual Plot: Gráfico de residuales vs. predicciones
ax2.scatter(y_pred_train, residuos_train, color='blue', marker='o', label='Residuales de Entrenamiento')
ax2.scatter(y_pred_test, residuos_test, color='green', marker='s', label='Residuales de Prueba')
ax2.axhline(y=0, color='red', linestyle='--', label='Línea de referencia')
ax2.set_xlabel('Predicciones')
ax2.set_ylabel('Residuales')
ax2.legend()

plt.tight_layout()
plt.show()
