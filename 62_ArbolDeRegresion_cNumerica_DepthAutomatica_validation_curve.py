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
x = df[['horsepower']]
y = df['price']

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

# Predice valores para nuevos datos
x_test = np.arange(0.0, 250.0, 1)[:, np.newaxis]
y_pred = regression_tree.predict(x_test)

# Grafica los resultados
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="Datos")
plt.plot(x_test, y_pred, color="cornflowerblue", lw=2, label="Predicción")
plt.xlabel("Potencia (HP)")
plt.ylabel("Precio")
plt.title("Árbol de Regresión (Max Depth = {})".format(optimal_depth))
plt.legend()
plt.show()

# Grafica la curva de validación
plt.figure()
plt.plot(max_depths, train_scores_mean, label="Puntuación de Entrenamiento", color="darkorange", lw=2)
plt.plot(max_depths, test_scores_mean, label="Puntuación de Validación", color="cornflowerblue", lw=2)
plt.xlabel("Max Depth")
plt.ylabel("Neg Mean Squared Error")
plt.title("Curva de Validación para Árbol de Regresión")
plt.legend()
plt.show()
