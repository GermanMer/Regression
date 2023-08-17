#CALCULA Y GRAFICA UNA REGRESIÓN POLINÓMICA MÚLTIPLE

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import numpy as np

# Cargar los datos y preparar el Data Frame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar las variables predictoras (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df[['price']]

# Transformar las características en una matriz polinómica
poly = PolynomialFeatures(degree=2)
x_poly = poly.fit_transform(x)

# Ajustar el modelo de regresión lineal a los datos polinómicos
model = LinearRegression()
model.fit(x_poly, y)

# Hacer una predicción
x_pred = pd.DataFrame({'horsepower': [220], 'curb-weight': [3700], 'engine-size': [270], 'width': [71]})
x_pred_poly = poly.transform(x_pred)
y_pred = model.predict(x_pred_poly)
print('Para un valor de horsepower', x_pred['horsepower'][0], ', curb-weight', x_pred['curb-weight'][0], ', engine-size', x_pred['engine-size'][0], 'y width', x_pred['width'][0], ', se predice un precio de', y_pred[0][0])

# Valores para graficar la curva polinómica
x_range_horsepower = np.linspace(x['horsepower'].min(), x['horsepower'].max(), num=100)
x_range_curb_weight = np.linspace(x['curb-weight'].min(), x['curb-weight'].max(), num=100)
x_range_engine_size = np.linspace(x['engine-size'].min(), x['engine-size'].max(), num=100)
x_range_width = np.linspace(x['width'].min(), x['width'].max(), num=100)

x_range_poly = poly.transform(np.column_stack((x_range_horsepower, x_range_curb_weight, x_range_engine_size, x_range_width)))
y_range_pred = model.predict(x_range_poly)

# Crear el gráfico
plt.figure(figsize=(10, 6))
plt.scatter(x['horsepower'], y['price'], color='blue', label='Datos reales')
plt.plot(x_range_horsepower, y_range_pred, color='orange', label='Regresión polinómica')
plt.scatter(x_pred['horsepower'], y_pred, color='magenta', marker='o', label='Predicción')
plt.annotate(f'${y_pred[0][0]:.2f}', (x_pred['horsepower'][0], y_pred[0]), textcoords="offset points", xytext=(0,10), ha='center', fontsize=12, color='magenta')
plt.xlabel('Variables Predictoras')
plt.ylabel('Precio')
plt.title('Regresión Polinómica Múltiple')
plt.legend()
plt.show()
