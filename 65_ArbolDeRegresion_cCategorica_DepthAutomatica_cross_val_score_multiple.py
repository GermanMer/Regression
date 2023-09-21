# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
import matplotlib.pyplot as plt

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['fuel-type', 'aspiration', 'make']]  # Utilizamos características categóricas
y = df['price']

# Convertir las columnas categóricas en variables numéricas utilizando one-hot encoding
x = pd.get_dummies(x, columns=['fuel-type', 'aspiration', 'make'],
                  prefix=['fuel', 'aspiration', 'make'])

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

# Entrenar el árbol de regresión con la profundidad óptima
regression_tree = DecisionTreeRegressor(max_depth=optimal_depth)
regression_tree.fit(x, y)

# Hacer predicciones
y_pred = regression_tree.predict(x)

# Calcula y muestra métricas de rendimiento adicionales
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Grafica los resultados
plt.figure(figsize=(10, 6))

# Datos reales
plt.scatter(y, y, s=20, edgecolor="black", c="darkorange", label="Datos Reales")

# Predicción del modelo
plt.scatter(y, y_pred, s=100, c="red", marker='x', label="Predicción")

plt.xlabel("Precio Real")
plt.ylabel("Predicción")
plt.title("Predicción de Precio vs. Precio Real")
plt.legend()
plt.grid(True)
plt.show()
