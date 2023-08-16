#CALCULA Y GRAFICA UNA REGRESIÓN LINEAL MÚLTIPLE

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

# Crear y entrenar el modelo de regresión lineal múltiple
lr_model = LinearRegression()
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
y = df[['price']]

lr_model.fit(x, y)

#Hacer una predicción
x_pred = pd.DataFrame({'horsepower': [220], 'curb-weight': [3700], 'engine-size': [270], 'width': [71]})
y_pred = lr_model.predict(x_pred)
print('Para un valor de horsepower', x_pred['horsepower'][0], ', curb-weight', x_pred['curb-weight'][0], ', engine-size', x_pred['engine-size'][0], 'y width', x_pred['width'][0], ', se predice un precio de', y_pred[0][0])

#Ver los interceptores y las pendientes
interceptor = lr_model.intercept_
print('El interceptor es:', interceptor)
slope = lr_model.coef_
print('Los coeficientes de las pendientes son:', slope)

#Gráfico
sns.regplot(x=df[['horsepower']], y='price', data=df, label='horsepower', color='blue')
sns.regplot(x=df[['curb-weight']], y='price', data=df, label='curb-weight', color='red')
sns.regplot(x=df[['engine-size']], y='price', data=df, label='engine-size', color='yellow')
sns.regplot(x=df[['width']], y='price', data=df, label='width', color='green')
plt.scatter(x_pred['horsepower'], y_pred, color='magenta', marker='o', label='Predicción')
plt.xlabel('Variables independientes')
plt.ylabel('Precio predicho')
plt.title('Regresión Lineal Múltiple')
plt.legend()
plt.show()
