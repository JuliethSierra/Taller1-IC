import pandas as pd

# Lee el archivo CSV
archivo_csv = 'filtrado_without_character.csv'  # Cambia esto a la ruta correcta

# Carga los datos desde el archivo CSV
dataframe = pd.read_csv(archivo_csv)

# Elimina las filas que contienen una o más columnas nulas
dataframe_sin_nulos = dataframe.dropna()

# Guarda el DataFrame sin filas nulas en un archivo CSV
archivo_csv_sin_nulos = 'filtrado_completo.csv'  # Cambia esto a la ruta donde deseas guardar el archivo sin nulos
dataframe_sin_nulos.to_csv(archivo_csv_sin_nulos, index=False)

print(f"Se han eliminado las filas con una o más columnas nulas y se ha guardado en {archivo_csv_sin_nulos}")
