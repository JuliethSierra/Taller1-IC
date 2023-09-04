
import pandas as pd

archivo_csv = 'filtrado_completo - copia.csv'  # Cambia esto a la ruta correcta

columnas_deseadas = ["ESTU_GENERO","FAMI_ESTRATOVIVIENDA","FAMI_TIENEINTERNET","FAMI_TIENECOMPUTADOR","COLE_BILINGUE","ESTU_GENERACION-E"]  # Cambia esto a las columnas que desees

# Lee el CSV y selecciona las columnas deseadas
dataframe = pd.read_csv(archivo_csv, usecols=columnas_deseadas)

# Imprime el DataFrame resultante
nuevo_archivo_csv = "filtrado_completo - copia_xd.csv"
dataframe.to_csv(nuevo_archivo_csv, index=False)
