#CALCULA Y GRAFICA UNA REGRESIÓN LINEAL SIMPLE

# Cargar los datos y preparar el Data Frame
import pandas as pd
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

#Crear y entrenar el modelo
from sklearn.linear_model import LinearRegression
lr_model = LinearRegression()
x = df[['horsepower']]
y = df[['price']]
lr_model.fit(x, y)

#Hacer una predicción
x_pred = pd.DataFrame({'horsepower': [220]}) #Valor para el cual queremos hacer la predicción
y_pred = lr_model.predict(x_pred) #Predicción
print('Para un valor de caballos de potencia de', x_pred['horsepower'][0], 'se predice un precio de', y_pred[0][0])

#Si queremos ver el valor del interceptor y la pendiente
interceptor = lr_model.intercept_
print('El interceptor es:', interceptor)
slope = lr_model.coef_
print('El coeficiente de la pendiente es:', slope)

#Graficar el modelo y la predicción
import seaborn as sns
import matplotlib.pyplot as plt
sns.regplot(x='horsepower', y='price', data=df, label='Datos reales')
plt.scatter(x_pred['horsepower'], y_pred, color='magenta', marker='o', label='Predicción')
plt.xlabel('Horsepower')
plt.ylabel('Precio')
plt.title('Regresión Lineal Simple')
plt.legend()
plt.show()
