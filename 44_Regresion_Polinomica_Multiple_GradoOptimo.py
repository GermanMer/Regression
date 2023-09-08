#BÚSQUEDA DEL GRADO POLINÓMICO OPTIMO PARA EVITAR OVERFITTING Y UNDERFITTING

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']

# Dividir los datos en conjuntos de entrenamiento y prueba
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

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

# Visualizar los resultados

# Valores
print('MSE Scores:', mse_scores)
print('R2 Scores:', r2_scores)
# Mejor grado
print('Mejor grado polinómico basado en MSE:', best_degree_mse)
print('Mejor grado polinómico basado en R2:', best_degree_r2)

# Gráfico
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), mse_scores, marker='o', label='MSE')
plt.plot(range(1, 11), r2_scores, marker='o', label='R²')
plt.xlabel('Grado Polinómico')
plt.ylabel('Puntuación')
plt.title('Evaluación de Diferentes Grados Polinómicos')
plt.legend()
plt.xticks(range(1, 11))
plt.grid()
plt.show()
