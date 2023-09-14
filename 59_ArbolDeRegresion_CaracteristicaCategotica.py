# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Seleccionar las características (x) y el objetivo (y)
x = df[['fuel-type']]
y = df['price']

# Convertir la columna categórica 'fuel-type' en variables numéricas utilizando one-hot encoding
x = pd.get_dummies(x, columns=['fuel-type'], prefix=['fuel'])

# Crear y entrenar el modelo de árbol de regresión
regressor = DecisionTreeRegressor()
regressor.fit(x, y)

# Hacer predicciones
y_pred = regressor.predict(x)

# Graficar los resultados
plt.scatter(x['fuel_gas'], y, color='blue', label='Gas')
plt.scatter(x['fuel_diesel'], y, color='green', label='Diesel')
plt.plot(x['fuel_gas'], y_pred, color='red', linewidth=2, label='Predicciones Gas')
plt.plot(x['fuel_diesel'], y_pred, color='orange', linewidth=2, label='Predicciones Diesel')
plt.xlabel('Tipo de Combustible')
plt.ylabel('Precio')
plt.legend()
plt.title('Árbol de Regresión para Predecir el Precio por Tipo de Combustible')
plt.show()
