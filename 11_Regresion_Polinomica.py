#CALCULA Y GRAFICA UNA REGRESIÓN POLINÓMICA

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower']]
y = df[['price']]

# Transformar las características en una matriz polinómica
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Ajustar el modelo de regresión lineal a los datos polinómicos
model = LinearRegression()
model.fit(x_poly, y)

# Predecir valores utilizando el modelo ajustado
x_pred = pd.DataFrame({'horsepower': [220]}) #Valor para el cual queremos hacer la predicción
x_pred_poly = poly.transform(x_pred)
y_pred = model.predict(x_pred_poly)

print('Para un valor de caballos de potencia de', x_pred['horsepower'][0], 'se predice un precio de', y_pred[0][0])

# Graficar el modelo y la predicción con una curva polinómica
# Valores para graficar la curva polinómica
x_range = np.linspace(df['horsepower'].min(), df['horsepower'].max(), num=100)
x_range_poly = poly.transform(x_range.reshape(-1, 1))
y_range_pred = model.predict(x_range_poly)
# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.scatter(x['horsepower'], y['price'], color='blue', label='Datos reales')
plt.plot(x_range, y_range_pred, color='orange', label='Regresión polinómica')
plt.scatter(x_pred['horsepower'], y_pred, color='magenta', marker='o', label='Predicción')
plt.xlabel('Horsepower')
plt.ylabel('Precio')
plt.title('Regresión Polinómica')
plt.legend()
plt.show()
