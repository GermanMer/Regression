#GRAFICA EL RESIDUAL PLOT PARA UNA REGRESIÓN LINEAL MÚLTIPLE

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Crear y entrenar el modelo de regresión lineal múltiple
lr_model = LinearRegression()
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df[['price']]

lr_model.fit(x, y)

# Hacer predicciones
y_pred = lr_model.predict(x)

# Calcular los residuales
residuos = y - y_pred

# Gráfico de residuales vs. predicciones
plt.scatter(y_pred, residuos, color='blue', marker='o', label='Residuales')
plt.axhline(y=0, color='red', linestyle='--', label='Línea de referencia')
plt.xlabel('Predicciones')
plt.ylabel('Residuales')
plt.title('Residual Plot de la Regresión Lineal Múltiple')
plt.legend()
plt.show()
