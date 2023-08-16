#GRAFICA EL RESIDUAL PLOT PARA UNA REGRESIÓN LINEAL MÚLTIPLE CON LAS VARIABLES INDEPENDIENTES NORMALIZADAS

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Crear el scaler y normalizar las variables independientes
scaler = StandardScaler()
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
x_scaled = scaler.fit_transform(x)

# Definir la vaiable dependiente
y = df[['price']]

# Crear y entrenar el modelo de regresión lineal múltiple
lr_model = LinearRegression()
lr_model.fit(x_scaled, y)

# Hacer predicciones
y_pred = lr_model.predict(x_scaled)

# Calcular los residuales
residuos = y - y_pred

#Gráfico de residuales vs. predicciones
plt.scatter(y_pred, residuos, color='blue', marker='o', label='Residuales')
plt.axhline(y=0, color='red', linestyle='--', label='Línea de referencia')
plt.xlabel('Predicciones')
plt.ylabel('Residuales')
plt.title('Residual Plot de la Regresión Lineal Múltiple con Variables Normalizadas')
plt.legend()
plt.show()
