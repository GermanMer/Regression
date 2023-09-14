# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Seleccionar las características (x) y el objetivo (y)
x = df[['fuel-type']]
y = df['price']

# Convertir la columna categórica 'fuel-type' en variables numéricas utilizando one-hot encoding
x = pd.get_dummies(x, columns=['fuel-type'], prefix=['fuel'])

# Definir una lista de valores para max_depth que deseas evaluar
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Crear listas para almacenar los resultados de la validación cruzada
cv_scores = []

# Realizar la validación cruzada para cada valor de max_depth
for depth in max_depths:
    regression_tree = DecisionTreeRegressor(max_depth=depth)
    scores = cross_val_score(regression_tree, x, y, cv=5, scoring='neg_mean_squared_error')
    cv_scores.append(-np.mean(scores))
    print(f"max_depth={depth}: Mean Squared Error={-np.mean(scores)}")

# Encontrar el valor óptimo de max_depth
optimal_depth = max_depths[np.argmin(cv_scores)]
print("La mejor profundidad (max_depth) es:", optimal_depth)

# Entrena el árbol de regresión con la profundidad óptima
regression_tree = DecisionTreeRegressor(max_depth=optimal_depth)
regression_tree.fit(x, y)

# Calcula y muestra métricas de rendimiento adicionales
y_pred = regression_tree.predict(x)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Graficar las predicciones del árbol de regresión
plt.scatter(x['fuel_gas'], y, color='blue', label='Gas')
plt.scatter(x['fuel_diesel'], y, color='green', label='Diesel')
plt.plot(x['fuel_gas'], y_pred, color='red', linewidth=2, label='Predicciones Gas')
plt.plot(x['fuel_diesel'], y_pred, color='orange', linewidth=2, label='Predicciones Diesel')
plt.xlabel('Tipo de Combustible')
plt.ylabel('Precio')
plt.legend()
plt.title('Árbol de Regresión para Predecir el Precio por Tipo de Combustible')
plt.show()
