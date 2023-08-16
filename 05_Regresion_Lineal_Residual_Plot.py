#GRAFICA EL RESIDUAL PLOT PARA UNA REGRESIÓN LINEAL SIMPLE

# Cargar los datos y preparar el Data Frame
import pandas as pd
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

#Crear y entrenar el modelo
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
x = df[['horsepower']]
y = df[['price']]
lr_model.fit(x, y)

#Hacer predicciones
y_pred = lr_model.predict(x)

# Calcular los residuales
residuos = y - y_pred

#Graficar el modelo y la predicción
import matplotlib.pyplot as plt
plt.scatter(y_pred, residuos, color='blue', marker='o', label='Residuales')
plt.axhline(y=0, color='red', linestyle='--', label='Línea de referencia')
plt.xlabel('Predicciones')
plt.ylabel('Residuales')
plt.title('Residual Plot de la Regresión Lineal Simple')
plt.legend()
plt.show()
