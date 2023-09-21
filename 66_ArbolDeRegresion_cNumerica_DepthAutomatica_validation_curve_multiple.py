# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import validation_curve, train_test_split
import matplotlib.pyplot as plt

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['horsepower', 'engine-size', 'curb-weight']]  # Usamos múltiples características
y = df['price']

# Dividir los datos en conjuntos de entrenamiento y prueba (por ejemplo, 80% entrenamiento, 20% prueba)
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Definir una lista de valores para max_depth que deseas evaluar
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Utilizar validation_curve para calcular las puntuaciones de validación y entrenamiento para diferentes valores de max_depth
train_scores, test_scores = validation_curve(
    DecisionTreeRegressor(), x_train, y_train, param_name="max_depth", param_range=max_depths,
    cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

# Calcular la media del error cuadrado negativo para las puntuaciones de entrenamiento y prueba
train_mse_mean = -np.mean(train_scores, axis=1)
test_mse_mean = -np.mean(test_scores, axis=1)

# Encontrar el valor óptimo de max_depth
optimal_depth = max_depths[np.argmin(test_mse_mean)]
print("La mejor profundidad (max_depth) es:", optimal_depth)

# Entrenar el árbol de regresión con la profundidad óptima en el conjunto de entrenamiento
regression_tree = DecisionTreeRegressor(max_depth=optimal_depth)
regression_tree.fit(x_train, y_train)

# Hacer predicciones en el conjunto de prueba
y_pred = regression_tree.predict(x_test)

# Calcular y mostrar métricas de rendimiento en el conjunto de prueba
from sklearn.metrics import mean_absolute_error, r2_score

mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"Mean Absolute Error en el conjunto de prueba: {mae}")
print(f"R² Score en el conjunto de prueba: {r2}")

# Graficar los resultados
plt.figure(figsize=(10, 6))

# Datos reales en el conjunto de prueba
plt.scatter(y_test, y_test, s=20, edgecolor="black", c="darkorange", label="Datos Reales (Prueba)")

# Predicción del modelo en el conjunto de prueba
plt.scatter(y_pred, y_pred, s=100, c="red", marker='x', label="Predicciones (Prueba)")

plt.xlabel("Precio Real")
plt.ylabel("Predicción")
plt.title("Predicción de Precio vs. Precio Real (Conjunto de Prueba)")
plt.legend()
plt.grid(True)
plt.show()
