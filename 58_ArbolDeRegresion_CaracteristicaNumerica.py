# Importa las bibliotecas necesarias
import pandas as pd
import numpy as np
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las características x e y
x = df[['horsepower']]
y = df['price']

# Entrena el árbol de regresión
regression_tree = DecisionTreeRegressor(max_depth=2)
regression_tree.fit(x, y)

# Predice valores para nuevos datos
x_test = np.arange(0.0, 250.0, 1)[:, np.newaxis]  # Ajusta el rango de x_test según tus datos
y_pred = regression_tree.predict(x_test)

# Grafica los resultados
plt.figure()
plt.scatter(x, y, s=20, edgecolor="black", c="darkorange", label="Datos")
plt.plot(x_test, y_pred, color="cornflowerblue", lw=2, label="Predicción")
plt.xlabel("Potencia (HP)")
plt.ylabel("Precio")
plt.title("Árbol de Regresión")
plt.legend()
plt.show()
