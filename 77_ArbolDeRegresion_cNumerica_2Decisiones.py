import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['horsepower', 'width']]  # Agrega 'width' a las características
y = df['price']

# Entrena el árbol de regresión con la profundidad óptima
# Cambia max_depth a 2 para agregar una segunda decisión
regression_tree = DecisionTreeRegressor(max_depth=2, criterion='squared_error')
regression_tree.fit(x, y)

# Calcula y muestra métricas de rendimiento adicionales
y_pred = regression_tree.predict(x)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Predice valores para nuevos datos
x_test_horsepower = np.arange(0.0, 250.0, 1)[:, np.newaxis]
x_test_width = np.arange(0.0, 75.0, 1)[:, np.newaxis]
y_pred_horsepower = regression_tree.predict(np.column_stack((x_test_horsepower, np.ones_like(x_test_horsepower) * x['width'].mean())))
y_pred_width = regression_tree.predict(np.column_stack((np.ones_like(x_test_width) * x['horsepower'].mean(), x_test_width)))

# Grafica los resultados (ahora con etiquetas corregidas)
plt.figure()
plt.scatter(x['width'], y, s=20, edgecolor="black", c="darkorange", label="Datos (width)")
plt.scatter(x['horsepower'], y, s=20, edgecolor="black", c="blue", label="Datos (horsepower)")
plt.plot(x_test_width, y_pred_width, color="green", lw=2, label="Predicción (width promedio)")
plt.plot(x_test_horsepower, y_pred_horsepower, color="cornflowerblue", lw=2, label="Predicción (horsepower promedio)")
plt.xlabel("Ancho (Width) / Potencia (HP)")
plt.ylabel("Precio")
plt.title("Árbol de Regresión (Max Depth = 2)")  # Cambia el título para reflejar la profundidad
plt.legend()
plt.show()
