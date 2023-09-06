import pandas as pd
import re

# Lee el archivo CSV en un DataFrame
archivo_csv = "Taller/example/filas_sin_nulos.csv"
df = pd.read_csv(archivo_csv, low_memory=False)

# Modifica la columna "Aprobado" utilizando la función replace
df['FAMI_ESTRATOVIVIENDA'] = df['FAMI_ESTRATOVIVIENDA'].replace({'Sin Estrato': 0, 'Estrato 1':1, 'Estrato 2': 2, 'Estrato 3': 3, 'Estrato 4': 4, 'Estrato 5': 5, 'Estrato 6': 6})
df['ESTU_GENERO'] = df['ESTU_GENERO'].replace({'M': 1, 'F': 2})
df['FAMI_TIENEINTERNET'] = df['FAMI_TIENEINTERNET'].replace({'Si': 1, 'No': 0})
df['FAMI_TIENECOMPUTADOR'] = df['FAMI_TIENECOMPUTADOR'].replace({'Si': 1, 'No': 0})
df['COLE_BILINGUE'] = df['COLE_BILINGUE'].replace({'S': 1, 'N': 0})
df['ESTU_GENERACION-E'] = df['ESTU_GENERACION-E'].replace({'GENERACION E - EXCELENCIA NACIONAL': 1, 'GENERACION E - EXCELENCIA DEPARTAMENTAL':1, 'GENERACION E - GRATUIDAD': 1, 'NO': 0})
df['FAMI_PERSONASHOGAR'] = df['FAMI_PERSONASHOGAR'].replace({'1 a 2': 4, '3 a 4':3, '5 a 6': 2, '7 a 8': 1,'9 o más': 0})
df['FAMI_TIENESERVICIOTV'] = df['FAMI_TIENESERVICIOTV'].replace({'Si': 1, 'No': 0})


df['FAMI_EDUCACIONPADRE'] = df['FAMI_EDUCACIONPADRE'].replace({
    'Educación profesional completa': 8,
    'Secundaria (Bachillerato) completa': 4,
    'Técnica o tecnológica completa': 7,
    'No sabe': 0,
    'Secundaria (Bachillerato) incompleta': 3,
    'Primaria completa': 2,
    'Educación profesional incompleta': 6,
    'Postgrado': 9,
    'Técnica o tecnológica incompleta': 5,
    'No Aplica': 0,
    'Primaria incompleta': 1,
    'Ninguno': 0
})

# Formatea la variable FAMI_EDUCACIONMADRE
df['FAMI_EDUCACIONMADRE'] = df['FAMI_EDUCACIONMADRE'].replace({
    'Educación profesional completa': 8,
    'Secundaria (Bachillerato) completa': 4,
    'Técnica o tecnológica completa': 7,
    'No sabe': 0,
    'Secundaria (Bachillerato) incompleta': 3,
    'Primaria completa': 2,
    'Educación profesional incompleta': 6,
    'Postgrado': 9,
    'Técnica o tecnológica incompleta': 5,
    'No Aplica': 0,
    'Primaria incompleta': 1,
    'Ninguno': 0
})

# Formatea la variable FAMI_NUMLIBROS
df['FAMI_NUMLIBROS'] = df['FAMI_NUMLIBROS'].replace({
    'MÁS DE 100 LIBROS': 3,
    '26 A 100 LIBROS': 2,
    '0 A 10 LIBROS': 0,
    '11 A 25 LIBROS': 1
})

# Formatea la variable ESTU_DEDICACIONLECTURADIARIA
df['ESTU_DEDICACIONLECTURADIARIA'] = df['ESTU_DEDICACIONLECTURADIARIA'].replace({
    'Entre 30 y 60 minutos': 2,
    'No leo por entretenimiento': 0,
    '30 minutos o menos': 1,
    'Entre 1 y 2 horas': 3,
    'Más de 2 horas': 4
})

# Formatea la variable ESTU_DEDICACIONINTERNET
df['ESTU_DEDICACIONINTERNET'] = df['ESTU_DEDICACIONINTERNET'].replace({
    'Entre 30 y 60 minutos': 2,
    'Más de 3 horas': 0,
    'Entre 1 y 3 horas': 1,
    'No Navega Internet': 4,
    '30 minutos o menos': 3
})

# Formatea la variable ESTU_HORASSEMANATRABAJA
df['ESTU_HORASSEMANATRABAJA'] = df['ESTU_HORASSEMANATRABAJA'].replace({
    'Menos de 10 horas': 3,
    '0': 4,  # Aquí asumo que '0' significa cero horas trabajadas
    'Más de 30 horas': 0,
    'Entre 11 y 20 horas': 2,
    'Entre 21 y 30 horas': 1
})


# Genera un nuevo archivo CSV con las modificaciones
nuevo_archivo_csv = "Taller/example/data_formateada_k.csv"
df.to_csv(nuevo_archivo_csv, index=False)

print(f"Valores con caracteres especiales o vacíos modificados y columna 'FAMI_TIENEINTERNET' modificada, y guardados en el archivo '{nuevo_archivo_csv}'.")

#One hot encoding
#MinMaxS