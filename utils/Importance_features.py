import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

# Lee el archivo CSV
archivo_csv = 'data\data_finales.csv'  # Cambia esto a la ruta correcta

# Carga los datos desde el archivo CSV
dataframe = pd.read_csv(archivo_csv)

# Divide los datos en características (X) y la variable objetivo (y)
X = dataframe.drop('ESTU_GENERACION-E', axis=1)  # Ajusta la columna objetivo
y = dataframe['ESTU_GENERACION-E']  # Ajusta la columna objetivo

# Crea el modelo de RandomForestClassifier
modelo = RandomForestClassifier()

# Entrena el modelo
modelo.fit(X, y)

# Obtiene la importancia de características
importancias = modelo.feature_importances_

# Crea un gráfico de barras para visualizar la importancia de características
plt.figure(figsize=(10, 6))
plt.bar(range(len(importancias)), importancias)
plt.xticks(range(len(importancias)), X.columns, rotation=90)
plt.xlabel('Características')
plt.ylabel('Importancia')
plt.title('Análisis de Importancia de Características')
plt.tight_layout()
plt.show()
