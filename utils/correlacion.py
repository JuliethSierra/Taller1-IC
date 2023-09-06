import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Lee el archivo CSV
archivo_csv = 'data\data_finales.csv'  # Cambia esto a la ruta correcta

# Carga los datos desde el archivo CSV
dataframe = pd.read_csv(archivo_csv)

# Calcula la matriz de correlación
matriz_correlacion = dataframe.corr()

# Crea un mapa de calor para visualizar la matriz de correlación
plt.figure(figsize=(10, 8))
sns.heatmap(matriz_correlacion, annot=True, cmap='coolwarm', center=0)
plt.title('Diagrama de Correlación')
plt.show()
