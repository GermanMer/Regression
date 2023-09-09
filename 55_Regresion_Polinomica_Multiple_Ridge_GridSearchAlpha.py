#BUSCA EL MEJOR VALOR PARA EL PARÁMETRO ALPHA DE LA REGRESIÓN DE RIDGE Y APLICA REGRESION DE RIDGE A UN MODELO DE REGRESIÓN POLINÓMICA

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']


###BUSCAR EL MEJOR VALOR DE ALPHA###
# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear un modelo de Ridge
model = Ridge()

# Definir los parámetros y valores a explorar en GridSearch
parametros = [{'alpha': [0.001, 0.01, 0.1, 1, 10, 100, 1000, 10000, 100000, 1000000]}]

# Crear el objeto GridSearchCV
grid = GridSearchCV(model, parametros, cv=5)

# Entrenar el objeto GridSearchCV
grid.fit(x_train, y_train)

# Imprimir el mejor valor de alpha encontrado por GridSearch
best_alpha = grid.best_estimator_.alpha
print("Mejor valor de alpha encontrado por GridSearch:", best_alpha)


###APLICAR REGRESION DE RIDGE###
# Transformar las características en una matriz polinómica
degree = 2
poly = PolynomialFeatures(degree=degree)
x_train_poly = poly.fit_transform(x_train)
x_test_poly = poly.transform(x_test)

# Crear un modelo de Ridge con el mejor valor de alpha
model = Ridge(alpha=best_alpha)

# Entrenar el modelo en los datos polinómicos
model.fit(x_train_poly, y_train)

# Hacer predicciones en el conjunto de entrenamiento y prueba
y_pred_train = model.predict(x_train_poly)
y_pred_test = model.predict(x_test_poly)

# Calcular el MSE en los conjuntos de entrenamiento y prueba
mse_train = mean_squared_error(y_train, y_pred_train)
mse_test = mean_squared_error(y_test, y_pred_test)
print("Mean Squared Error (Train):", mse_train)
print("Mean Squared Error (Test):", mse_test)

# Calcular el R Squared en los conjuntos de entrenamiento y prueba
r2_train = r2_score(y_train, y_pred_train)
r2_test = r2_score(y_test, y_pred_test)
print("R Squared (Train):", r2_train)
print("R Squared (Test):", r2_test)
