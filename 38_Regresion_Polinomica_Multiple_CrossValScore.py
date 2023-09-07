#REALIZA VALIDACIÓN CRUZADA PARA UNA REGRESIÓN POLINÓMICA MÚLTIPLE

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import make_scorer, mean_squared_error
from sklearn.pipeline import make_pipeline

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']

# Crear un modelo de regresión polinómica de grado 2
model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

# Definir la métrica de evaluación (error cuadrático medio)
scoring = make_scorer(mean_squared_error, greater_is_better=False)

# Realizar validación cruzada con 5 pliegues (folds)
# cross_val_score devuelve una lista de puntuaciones de rendimiento en cada pliegue
scores = cross_val_score(model, x, y, cv=5, scoring=scoring)

# Convertir las puntuaciones negativas en positivas
positive_scores = -scores

# Imprimir las puntuaciones de rendimiento en cada pliegue
print("Puntuaciones de Validación Cruzada:", positive_scores)
print("Promedio de Puntuaciones:", positive_scores.mean())
