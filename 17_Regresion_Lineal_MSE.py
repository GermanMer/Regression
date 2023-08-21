#CALCULA EL MSE PARA UNA REGRESIÓN LINEAL SIMPLE

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

#Calcular el MSE
from sklearn.metrics import mean_squared_error
mse = mean_squared_error(y, y_pred)
print("Mean Squared Error:", mse)
