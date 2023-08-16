#CREA UN HEATMAP DE LA MATRIZ DE CORRELACIÓN DE UN DATAFRAME

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos y preparar el DataFrame
df = pd.read_csv(r'D:\Germán\Desktop\Python Files\automobile.csv')
df = df.dropna(axis=0)

#Crea y muestra el Heatmap de la matriz de correlación
sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
plt.title('Heatmap de la matriz de correlación')
plt.show()
