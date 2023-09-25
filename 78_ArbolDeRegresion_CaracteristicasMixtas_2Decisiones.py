import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_absolute_error, r2_score
import matplotlib.pyplot as plt

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Codificar 'fuel-type' como características binarias (One-Hot Encoding)
df = pd.get_dummies(df, columns=['fuel-type'], prefix=['fuel'])

# Separar las características x e y
x = df[['horsepower', 'fuel_gas', 'fuel_diesel']]  # Agrega 'fuel' a las características
y = df['price']

# Entrena el árbol de regresión con la profundidad óptima
regression_tree = DecisionTreeRegressor(max_depth=2, criterion='squared_error')
regression_tree.fit(x, y)

# Calcula y muestra métricas de rendimiento adicionales
y_pred = regression_tree.predict(x)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Predice valores para nuevos datos manteniendo 'fuel' constante
x_test_horsepower = np.arange(0.0, 250.0, 1)[:, np.newaxis]
x_test_fuel_gas = np.ones_like(x_test_horsepower)  # 'fuel_gas' se establece en 1 para todas las muestras de prueba
x_test_fuel_diesel = np.zeros_like(x_test_horsepower)  # 'fuel_diesel' se establece en 0 para todas las muestras de prueba
y_pred_horsepower_gas = regression_tree.predict(np.column_stack((x_test_horsepower, x_test_fuel_gas, x_test_fuel_diesel)))

# Grafica los resultados con las columnas corregidas
plt.figure()
plt.scatter(df[df['fuel_gas'] == 1]['horsepower'], df[df['fuel_gas'] == 1]['price'], s=20, edgecolor="black", c="blue", label="Datos (gas)")
plt.plot(x_test_horsepower, y_pred_horsepower_gas, color="cornflowerblue", lw=2, label="Predicción (gas)")
plt.scatter(df[df['fuel_diesel'] == 1]['horsepower'], df[df['fuel_diesel'] == 1]['price'], s=20, edgecolor="black", c="red", label="Datos (diesel)")
plt.xlabel("Potencia (HP)")
plt.ylabel("Precio")
plt.title("Árbol de Regresión (Max Depth = 2)")
plt.legend()
plt.show()
