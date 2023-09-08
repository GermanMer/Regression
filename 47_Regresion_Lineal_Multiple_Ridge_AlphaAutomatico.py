#BUSCA EL MEJOR VALOR PARA EL PARÁMETRO ALPHA DE LA REGRESIÓN DE RIDGE Y APLICA REGRESION DE RIDGE A UN MODELO DE REGRESIÓN LINEAL MÚLTIPLE

import numpy as np
from sklearn.model_selection import cross_val_score
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']


###BUSCAR EL MEJOR VALOR DE ALPHA###
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


###APLICAR REGRESION DE RIDGE###
# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

# Crear un modelo de Ridge con un valor de alpha (parámetro de regularización)
alpha = best_alpha
model = Ridge(alpha=alpha)

# Entrenar el modelo
model.fit(x_train, y_train)

# Hacer predicciones en el conjunto de entrenamiento y prueba
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

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
