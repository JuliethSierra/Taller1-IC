import pandas as pd

# Lee el archivo CSV
archivo_csv = 'data_clean - copia.csv'  # Cambia esto a la ruta correcta

# Carga los datos desde el archivo CSV
dataframe = pd.read_csv(archivo_csv)

# Reemplaza los guiones "-" por una cadena vacía en todo el DataFrame
dataframe.replace("-", "", inplace=True)

# Guarda el DataFrame modificado en un archivo CSV
archivo_csv_modificado = 'clean_character.csv'  # Cambia esto a la ruta donde deseas guardar el archivo modificado
dataframe.to_csv(archivo_csv_modificado, index=False)

print(f"Se han reemplazado los guiones por una cadena vacía y se ha guardado en {archivo_csv_modificado}")
