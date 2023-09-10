# DIVIDE EL DATASET EN TRAIN Y TEST SETS, HACE PREDICCIONES USANDO REGRESIÓN RIDGE MÚLTIPLE CON VARIABLES NORMALIZADAS, CALCULA EL MSE Y R2 Y GRAFICA EL RESULTADO JUNTO CON EL RESIDUAL PLOT Y EL DISTRIBUTION PLOT.

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
import seaborn as sns
import matplotlib.pyplot as plt

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Crear el scaler y normalizar las variables independientes
scaler = StandardScaler()
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
x_scaled = scaler.fit_transform(x)

# Definir la vaiable dependiente
y = df[['price']]


###BUSCAR EL MEJOR VALOR DE ALPHA###
# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x_scaled, y, test_size=0.2, random_state=42)

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
# Crear un modelo de Ridge con el mejor valor de alpha
model = Ridge(alpha=best_alpha)

# Entrenar el modelo
model.fit(x_train, y_train)

# Hacer predicciones en el conjunto de entrenamiento y prueba
y_pred_train = model.predict(x_train)
y_pred_test = model.predict(x_test)

# Interceptor y pendiente
interceptor = model.intercept_
print('El interceptor es:', interceptor)
slope = model.coef_
print('El coeficiente de la pendiente es:', slope)

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

# Calcular los residuales
residuos_train = y_train.values - y_pred_train
residuos_test = y_test.values - y_pred_test

# Gráfico 1
fig, axes = plt.subplots(nrows=1, ncols=4, figsize=(20, 5))

# Variables independientes para la visualización
independent_vars = ['horsepower', 'curb-weight', 'engine-size', 'width']

# Hacer subgráficos para cada variable independiente
for i, var in enumerate(independent_vars):
    axes[i].scatter(x_train[:, i], y_train, color='blue', label='Training Data')
    axes[i].scatter(x_test[:, i], y_test, color='green', label='Test Data')

    # Ajustar la línea de regresión en ambos gráficos de entrenamiento y prueba
    model.fit(x_train[:, [i]], y_train)
    y_pred_train_var = model.predict(x_train[:, [i]])
    y_pred_test_var = model.predict(x_test[:, [i]])

    # Trazar la línea de regresión en ambos gráficos de entrenamiento y prueba
    axes[i].plot(x_train[:, i], y_pred_train_var, color='orange', label='Regresión lineal')
    axes[i].plot(x_test[:, i], y_pred_test_var, color='red', label='Regresión lineal (Test)')

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
