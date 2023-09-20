# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

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
train_scores, test_scores = validation_curve(
    DecisionTreeRegressor(),
    x,
    y,
    param_name="max_depth",
    param_range=max_depths,
    cv=5,
    scoring="neg_mean_squared_error"
)

train_scores_mean = -np.mean(train_scores, axis=1)
test_scores_mean = -np.mean(test_scores, axis=1)

# Encontrar el valor óptimo de max_depth
optimal_depth = max_depths[np.argmin(test_scores_mean)]
print("La mejor profundidad (max_depth) es:", optimal_depth)

# Entrena el árbol de regresión con la profundidad óptima
regression_tree = DecisionTreeRegressor(max_depth=optimal_depth)
regression_tree.fit(x, y)

# Hace predicciones
y_pred = regression_tree.predict(x)

# Grafica los resultados
# Graficar las curvas de validación
plt.figure(figsize=(10, 6))
plt.plot(max_depths, train_scores_mean, marker='o', label='Training Score')
plt.plot(max_depths, test_scores_mean, marker='o', label='Validation Score')
plt.xlabel('Profundidad del Árbol (max_depth)')
plt.ylabel('Negativo del Error Cuadrático Medio (MSE)')
plt.title('Curva de Validación para el Árbol de Regresión')
plt.legend()
plt.grid(True)
plt.show()

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
