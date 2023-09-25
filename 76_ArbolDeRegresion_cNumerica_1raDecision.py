import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, r2_score

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['horsepower']]
y = df['price']

# Entrena el árbol de regresión con la profundidad óptima
# Cambia max_depth a 1 para que la primera decisión sea sobre 100 caballos de fuerza
regression_tree = DecisionTreeRegressor(max_depth=1, criterion='squared_error')
regression_tree.fit(x, y)

# Calcula y muestra métricas de rendimiento adicionales
y_pred = regression_tree.predict(x)
mae = mean_absolute_error(y, y_pred)
r2 = r2_score(y, y_pred)
print(f"Mean Absolute Error: {mae}")
print(f"R² Score: {r2}")

# Predice valores para nuevos datos
x_test = np.arange(0.0, 250.0, 1)[:, np.newaxis]
y_pred = regression_tree.predict(x_test)

# Grafica los resultados
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="Datos")
plt.plot(x_test, y_pred, color="cornflowerblue", lw=2, label="Predicción")
plt.xlabel("Potencia (HP)")
plt.ylabel("Precio")
plt.title("Árbol de Regresión (Max Depth = 1)")  # Cambia el título para reflejar la profundidad
plt.legend()
plt.show()
