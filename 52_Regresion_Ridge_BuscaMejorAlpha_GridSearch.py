#BUSCA EL MEJOR ALPHA USANDO GRIDSEARCH PARA UNA REGRESION DE RIDGE LINEAL MÚLTIPLE

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Separar la variable independiente predictora (x) y la variable dependiente objetivo (y)
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df['price']

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

# Imprimir el mejor estimador (modelo) encontrado por GridSearch
print("Mejor estimador encontrado por GridSearch:", grid.best_estimator_)

# Obtener los resultados de la búsqueda en una variable
scores = grid.cv_results_

# Imprimir los resultados de GridSearch
for param, mean_val in zip(scores['params'], scores['mean_test_score']):
    print(param, 'R2 en datos de prueba:', mean_val)
