# BUSCA EL MEJOR GRADO POLINÓMICO y DIVIDE EL DATASET EN TRAIN Y TEST SETS, HACE PREDICCIONES USANDO REGRESIÓN POLINÓMICA MÚLTIPLE RIDGE, CALCULA EL MSE Y R2 Y GRAFICA EL RESULTADO JUNTO CON EL RESIDUAL PLOT Y EL DISTRIBUTION PLOT.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import Ridge
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.model_selection import GridSearchCV
import seaborn as sns
import matplotlib.pyplot as plt

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


###BUSCAR EL GRADO POLINÓMICO ÓPTIMO###
# Inicializar listas para almacenar los resultados de validación cruzada
mse_scores = []
r2_scores = []

# Probar diferentes grados polinómicos
for degree in range(1, 11):
    poly = PolynomialFeatures(degree=degree)
    x_train_poly = poly.fit_transform(x_train)

    model = LinearRegression()

    # Realizar validación cruzada con 5 pliegues (folds)
    mse_scores_fold = -cross_val_score(model, x_train_poly, y_train, cv=5, scoring='neg_mean_squared_error')
    mse_scores.append(np.mean(mse_scores_fold))

    r2_scores_fold = cross_val_score(model, x_train_poly, y_train, cv=5, scoring='r2')
    r2_scores.append(np.mean(r2_scores_fold))

# Encontrar el mejor grado polinómico
best_degree_mse = np.argmin(mse_scores) + 1  # Sumar 1 para ajustar el índice
best_degree_r2 = np.argmax(r2_scores) + 1    # Sumar 1 para ajustar el índice

# Valores
print('MSE Scores:', mse_scores)
print('R2 Scores:', r2_scores)
# Mejor grado
print('Mejor grado polinómico basado en MSE:', best_degree_mse)
print('Mejor grado polinómico basado en R2:', best_degree_r2)


###APLICAR REGRESION DE RIDGE###
poly = PolynomialFeatures(degree=best_degree_mse)
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

# Realizar validación cruzada con 5 pliegues (folds)
# cross_val_score devuelve una lista de puntuaciones de rendimiento en cada pliegue
scores = cross_val_score(model, x, y, cv=5, scoring='neg_mean_squared_error')

# Convertir las puntuaciones negativas en positivas
positive_scores = -scores

# Imprimir las puntuaciones de rendimiento en cada pliegue
print("Puntuaciones de Validación Cruzada:", positive_scores)
print("Promedio de Puntuaciones:", positive_scores.mean())

# cross_val_predict devuelve un array de predicciones para cada instancia
predictions = cross_val_predict(model, x, y, cv=5)

# Imprimir las predicciones
print("Predicciones:", predictions)

# Calcular los residuales
residuos_train = y_train.values - y_pred_train
residuos_test = y_test.values - y_pred_test

# Gráfico 1
# Crear una figura con varios subgráficos
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

# Variables independientes para la visualización
independent_vars = ['horsepower', 'curb-weight', 'engine-size', 'width']

# Hacer subgráficos para cada variable independiente
for i, var in enumerate(independent_vars):
    # Scatter plot de los datos de entrenamiento y prueba
    axes[i].scatter(x_train[var], y_train, color='blue', label='Training Data')
    axes[i].scatter(x_test[var], y_test, color='green', label='Test Data')

    # Ajustar la regresión polinómica en los datos de entrenamiento
    poly_features = poly.fit_transform(x_train[[var]])
    model.fit(poly_features, y_train)
    y_pred_train_var = model.predict(poly_features)

    # Trazar la curva de regresión polinómica en los gráficos de entrenamiento y prueba
    x_range = np.linspace(min(x_train[var]), max(x_train[var]), num=100)
    x_range_poly = poly.transform(x_range.reshape(-1, 1))
    y_range_pred = model.predict(x_range_poly)

    axes[i].plot(x_range, y_range_pred, color='orange', label='Regresión polinómica')
    axes[i].set_xlabel(var)
    axes[i].set_ylabel('Price')
    axes[i].legend()

plt.tight_layout()

# Gráfico 2
fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(15, 5))

# Distribution Plot
sns.distplot(y, hist=False, color='red', label='Datos reales', ax=ax1)
sns.distplot(y_pred_train, hist=False, color='blue', label='Predicciones de Entrenamiento', ax=ax1)
sns.distplot(y_pred_test, hist=False, color='green', label='Predicciones de Prueba', ax=ax1)
ax1.set_xlabel('Precio')
ax1.set_ylabel('Densidad')
ax1.legend()

# Residual Plot: Gráfico de residuales vs. predicciones
ax2.scatter(y_pred_train, residuos_train, color='blue', marker='o', label='Residuales de Entrenamiento')
ax2.scatter(y_pred_test, residuos_test, color='green', marker='s', label='Residuales de Prueba')
ax2.axhline(y=0, color='red', linestyle='--', label='Línea de referencia')
ax2.set_xlabel('Predicciones')
ax2.set_ylabel('Residuales')
ax2.legend()

plt.tight_layout()
plt.show()
