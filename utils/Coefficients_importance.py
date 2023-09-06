import pandas as pd
import matplotlib.pyplot as plt

# Carga el archivo CSV en un DataFrame de Pandas
data = pd.read_csv('data\data_finales.csv')

# Muestra las primeras filas del DataFrame para verificar la carga de datos
print(data.head())

# Calcula la matriz de correlación de Pearson
correlation_matrix = data.corr()

# Extrae los coeficientes de correlación de Pearson con respecto a una variable objetivo (por ejemplo, 'target')
correlation_with_target = correlation_matrix['ESTU_GENERACION-E']  # Reemplaza 'target' con el nombre de tu variable objetivo

# Crea un gráfico de barras para visualizar los coeficientes de correlación con respecto a la variable objetivo
plt.figure(figsize=(10, 6))
correlation_with_target.plot(kind='bar', color='Blue', alpha=0.7)

# Personaliza el gráfico
plt.title('Importancia de los Coeficientes de Correlación con respecto a la Variable Objetivo')
plt.xlabel('Variables Independientes')
plt.ylabel('Coeficiente de Correlación de Pearson')

# Muestra el gráfico de barras
plt.show()
