# DIVIDE EL DATASET EN TRAIN Y TEST SETS, HACE PREDICCIONES USANDO REGRESIÓN LINEAL SIMPLE, CALCULA EL MSE Y R2 Y GRAFICA EL RESULTADO JUNTO CON EL RESIDUAL PLOT Y EL DISTRIBUTION PLOT.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower']]
y = df[['price']]

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear y entrenar el modelo
model = LinearRegression()
model.fit(x_train, y_train)

# Hacer predicciones
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Interceptor y pendiente
interceptor = model.intercept_
print('El interceptor es:', interceptor)
slope = model.coef_
print('El coeficiente de la pendiente es:', slope)

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

# Gráfico de dispersión y regresión lineal
plt.scatter(x_train, y_train, color='blue', label='Training Data')
plt.scatter(x_test, y_test, color='green', label='Test Data')
plt.plot(x_train, y_pred_train, color='orange', label='Regresión lineal')
plt.plot(x_test, y_pred_test, color='red', label='Regresión lineal (Test)')
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
