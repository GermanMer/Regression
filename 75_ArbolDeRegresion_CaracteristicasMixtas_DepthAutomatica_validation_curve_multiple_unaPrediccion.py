# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import validation_curve
import matplotlib.pyplot as plt

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['horsepower', 'curb-weight', 'aspiration', 'make']]  # Combinación de características numéricas y categóricas
y = df['price']

# Convertir las columnas categóricas en variables dummy (one-hot encoding)
x = pd.get_dummies(x, columns=['aspiration', 'make'], prefix=['aspiration', 'make'])

# Definir una lista de valores para max_depth que deseas evaluar
max_depths = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# Utilizar validation_curve para calcular las puntuaciones de validación y entrenamiento para diferentes valores de max_depth
train_scores, test_scores = validation_curve(
    DecisionTreeRegressor(), x, y, param_name="max_depth", param_range=max_depths,
    cv=5, scoring="neg_mean_squared_error", n_jobs=-1)

# Calcular la media del error cuadrado negativo para las puntuaciones de entrenamiento y prueba
train_mse_mean = -np.mean(train_scores, axis=1)
test_mse_mean = -np.mean(test_scores, axis=1)

# Encontrar el valor óptimo de max_depth
optimal_depth = max_depths[np.argmin(test_mse_mean)]
print("La mejor profundidad (max_depth) es:", optimal_depth)

# Entrenar el árbol de regresión con la profundidad óptima
regression_tree = DecisionTreeRegressor(max_depth=optimal_depth)
regression_tree.fit(x, y)

# Calcula y muestra métricas de rendimiento adicionales
from sklearn.metrics import mean_absolute_error, r2_score
y_pred = regression_tree.predict(x)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Predice valores para x_test
x_test = pd.DataFrame({'horsepower': [100], 'curb-weight': [2500], 'aspiration_std': [1], 'aspiration_turbo': [0], 'make_audi': [0], 'make_bmw': [1], 'make_chevrolet': [0], 'make_dodge': [0], 'make_honda': [0], 'make_jaguar': [0], 'make_mazda': [0],
       'make_mercedes-benz': [0], 'make_mitsubishi': [0], 'make_nissan': [0], 'make_peugot': [0], 'make_plymouth': [0],
       'make_porsche': [0], 'make_saab': [0], 'make_subaru': [0], 'make_toyota': [0], 'make_volkswagen': [0], 'make_volvo': [0]})

# Asegurarse de que las columnas en x_test estén en el mismo orden que en x
column_order = x.columns
x_test = x_test[column_order]

# Hace predicciones
y_pred = regression_tree.predict(x_test)

# Impresión de la predicción
print("Predicción de Precio:", y_pred)

# Grafica los resultados
plt.figure(figsize=(10, 6))

# Datos reales
plt.scatter(y, y, s=20, edgecolor="black", c="darkorange", label="Datos Reales")

# Predicción del modelo
plt.scatter(y_pred, y_pred, s=100, c="red", marker='x', label="Predicción")

plt.xlabel("Precio Real")
plt.ylabel("Predicción")
plt.title("Predicción de Precio vs. Precio Real")
plt.legend()
plt.grid(True)
plt.show()
