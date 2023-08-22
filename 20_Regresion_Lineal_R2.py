#CALCULA EL R SQUARED PARA UNA REGRESIÓN LINEAL SIMPLE

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

#Calcular el R Squared
from sklearn.metrics import r2_score
r2 = r2_score(y, y_pred)
print("R Squared:", r2)
