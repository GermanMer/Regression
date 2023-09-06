#REALIZA VALIDACIÓN CRUZADA PARA UNA REGRESIÓN LINEAL SIMPLE

import pandas as pd
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower']]
y = df['price']

# Crear un modelo de regresión lineal
model = LinearRegression()

# Realizar validación cruzada con 5 pliegues (folds)
# cross_val_score devuelve una lista de puntuaciones de rendimiento en cada pliegue
scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

# Convertir las puntuaciones negativas en positivas
positive_scores = -scores

# Imprimir las puntuaciones de rendimiento en cada pliegue
print("Puntuaciones de Validación Cruzada:", positive_scores)
print("Promedio de Puntuaciones:", positive_scores.mean())
