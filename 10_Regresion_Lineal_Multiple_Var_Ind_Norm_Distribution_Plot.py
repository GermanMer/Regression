#GRAFICA EL DISTRIBUTION PLOT PARA UNA REGRESIÓN LINEAL MÚLTIPLE CON VARIABLES INDEPENDIENTES NORMALIZADAS

# Cargar los datos y preparar el Data Frame
import pandas as pd
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

#Crear y entrenar el modelo
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
lr_model = LinearRegression()
x = df[['horsepower', 'curb-weight', 'engine-size', 'width']]
scaler = StandardScaler()
x_scaled = scaler.fit_transform(x)
y = df[['price']]
lr_model.fit(x_scaled, y)

#Hacer predicciones
y_pred = lr_model.predict(x_scaled)

#Graficar el modelo y la predicción
import seaborn as sns
import matplotlib.pyplot as plt
ax1 = sns.distplot(df[['price']], hist=False, color='r', label='Datos reales')
sns.distplot(y_pred, hist=False, color='b', label='Datos predichos', ax=ax1)
plt.xlabel('Precio')
plt.ylabel('Densidad')
plt.title('Distribution Plot de la Regresión Lineal Múltiple')
plt.legend()
plt.show()
