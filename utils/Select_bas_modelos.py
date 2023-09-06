import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

# Lee el archivo CSV
archivo_csv = 'data\data_finales.csv'  # Cambia esto a la ruta correcta

# Carga los datos desde el archivo CSV
dataframe = pd.read_csv(archivo_csv)

# Divide los datos en características (X) y la variable objetivo (y)
X = dataframe.drop('ESTU_GENERACION-E', axis=1)  # Ajusta la columna objetivo
y = dataframe['ESTU_GENERACION-E']  # Ajusta la columna objetivo

# Crea el modelo base
modelo_base = RandomForestClassifier()

# Crea el selector basado en modelos
selector = SelectFromModel(modelo_base)

# Realiza la selección de características
X_seleccionado = selector.fit_transform(X, y)

# Muestra las características seleccionadas
caracteristicas_seleccionadas = X.columns[selector.get_support()]
print("Características seleccionadas:\n", caracteristicas_seleccionadas)
