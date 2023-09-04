import pandas as pd
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression

# Lee el archivo CSV
archivo_csv = 'data_finales.csv'  # Cambia esto a la ruta correcta

# Carga los datos desde el archivo CSV
dataframe = pd.read_csv(archivo_csv)

# Divide los datos en características (X) y la variable objetivo (y)
X = dataframe.drop('ESTU_GENERACION-E', axis=1)  # Ajusta la columna objetivo
y = dataframe['ESTU_GENERACION-E']  # Ajusta la columna objetivo

# Crea el modelo base
modelo_base = LogisticRegression()

# Crea el selector RFE
selector_rfe = RFE(modelo_base, n_features_to_select=5)  # Ajusta el número de características deseadas

# Realiza la selección de características
X_rfe = selector_rfe.fit_transform(X, y)

# Muestra las características seleccionadas
caracteristicas_seleccionadas = X.columns[selector_rfe.support_]
print("Características seleccionadas:", caracteristicas_seleccionadas)
