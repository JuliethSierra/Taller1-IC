import pandas as pd
import re

# Lee el archivo CSV en un DataFrame
archivo_csv = "Saber_11__2019-2.csv"
df = pd.read_csv(archivo_csv, low_memory=False)

# Modifica la columna "Aprobado" utilizando la función replace
df['FAMI_ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].replace({'Sin Estrato': 0, 'Estrato 1':1, 'Estrato 2': 2, 'Estrato 3': 3, 'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6})
df['ESTU_GENERO'] = df['ESTU_GENERO'].replace({'M': 1, 'F': 2})
df['FAMI_TIENEINTERNET'] = df['FAMI_TIENEINTERNET'].replace({'Si': 1, 'No': 2})
df['FAMI_TIENECOMPUTADOR'] = df['FAMI_TIENECOMPUTADOR'].replace({'Si': 1, 'No': 2})
df['COLE_BILINGUE'] = df['COLE_BILINGUE'].replace({'S': 1, 'N': 2})

df['ESTU_GENERACION-E'] = df['ESTU_GENERACION-E'].replace({'GENERACION E - EXCELENCIA NACIONAL': 1, 'GENERACION E - EXCELENCIA DEPARTAMENTAL':1, 'GENERACION E - GRATUIDAD': 1, 'NO': 2})

# Genera un nuevo archivo CSV con las modificaciones
nuevo_archivo_csv = "data_finales.csv"
df.to_csv(nuevo_archivo_csv, index=False)

print(f"Valores con caracteres especiales o vacíos modificados y columna 'FAMI_TIENEINTERNET' modificada, y guardados en el archivo '{nuevo_archivo_csv}'.")



#One hot encoding
#MinMaxS