#REALIZA PREDICCIONES USANDO VALIDACIÓN CRUZADA PARA UNA REGRESIÓN LINEAL SIMPLE

import pandas as pd
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LinearRegression

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower']]
y = df['price']

# Crear un modelo de regresión lineal
model = LinearRegression()

# Realizar predicciones utilizando validación cruzada con 5 pliegues (folds)
# cross_val_predict devuelve un array de predicciones para cada instancia
predictions = cross_val_predict(model, x, y, cv=5)

# Imprimir las predicciones
print("Predicciones:", predictions)
