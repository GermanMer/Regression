#GRAFICA EL DISTRIBUTION PLOT PARA UNA REGRESIÓN LINEAL SIMPLE

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

#Hacer predicciones
y_pred = lr_model.predict(x)

#Graficar el modelo y la predicción
import seaborn as sns
import matplotlib.pyplot as plt
ax1 = sns.distplot(df[['price']], hist=False, color='r', label='Datos reales')
sns.distplot(y_pred, hist=False, color='b', label='Datos predichos', ax=ax1)
plt.xlabel('Precio')
plt.ylabel('Densidad')
plt.title('Distribution Plot de la Regresión Lineal Simple')
plt.legend()
plt.show()
