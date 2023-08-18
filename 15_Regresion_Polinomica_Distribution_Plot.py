#GRAFICA EL DISTRIBUTION PLOT PARA UNA REGRESIÓN POLINÓMICA

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt
import seaborn as sns

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

# Calcular los valores predichos y los residuos
y_pred = model.predict(x_poly)
residuals = y - y_pred

# Crear el gráfico de distribución de residuos en el estilo deseado
plt.figure(figsize=(8, 6))
ax1 = sns.distplot(y, hist=False, color='r', label='Datos reales')
sns.distplot(y_pred, hist=False, color='b', label='Datos predichos', ax=ax1)
plt.title('Distribution Plot de la Regresión Polinómica')
plt.xlabel('Valores de Precio')
plt.ylabel('Densidad')
plt.legend()
plt.show()
