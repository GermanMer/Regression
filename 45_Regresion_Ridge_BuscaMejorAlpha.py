#BUSCA EL MEJOR VALOR PARA EL PARÁMETRO ALPHA DE LA REGRESIÓN DE RIDGE

import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import Ridge
import pandas as pd

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']

# Lista de valores de alpha para probar
alphas = np.logspace(-6, 6, 13)

# Inicializar una lista para almacenar las puntuaciones de validación cruzada
scores = []

# Realizar validación cruzada para cada valor de alpha
for alpha in alphas:
    model = Ridge(alpha=alpha)
    cv_scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')
    scores.append(-np.mean(cv_scores))  # Multiplicar por -1 para obtener valores positivos

# Encontrar el valor de alpha con el mejor rendimiento
best_alpha = alphas[np.argmax(scores)]
print("Mejor valor de alpha:", best_alpha)
