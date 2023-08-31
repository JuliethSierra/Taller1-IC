# -*- coding: utf-8 -*-
"""SVM.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19AKPq5kXOPhEQdokSab9lZsDBgyNvfWo

#Librerías a usar
"""

# Librerías a usar
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sb

#Preprocesamiento y modelado
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix

import warnings
warnings.filterwarnings('ignore')

"""# Carga de datos y preprocesamiento"""

# Carga de datos
url = '/content/drive/MyDrive/2023-II/Inteligencia_Computacional/data_finales.csv' #Cambiar ruta a CSV con valores normalizados
# url = 'https://www.datos.gov.co/resource/ynam-yc42.csv' # Tabla de valores sin normalizar
dataset = pd.read_csv(url)

# Dimensiones y registros del dataset
print(dataset.shape)
# Primeros registros
dataset.head()

#Verificación de tipos de cada columna y completitud de datos
print(dataset.info())
sb.heatmap(dataset.isnull(), cmap='rainbow')

"""# Creación y entrenamiento del modelo de SVM"""

# Definición de datos X y Y
columns = ['ESTU_GENERO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'COLE_BILINGUE'] # Columnas de datos normalizados
# columns = ['estu_genero', 'fami_estratovivienda', 'fami_tieneinternet', 'fami_tienecomputador', 'cole_bilingue'] # Columnas de datos sin normalizar
# y = dataset['estu_generacion-e'] # Columna de datos sin normalizar
X = dataset.drop(columns, axis=1)
y = dataset['ESTU_GENERACION-E'] # Columna de datos normalizados

# Entrenamiento de modelo
X_train, X_test, y_train, y_test = train_test_split(X,
                                                    y,
                                                    test_size=0.25,
                                                    random_state=42)

# Implementación de modelo
model = SVC()
model.fit(X_train, y_train)

"""# Evaluación del modelo"""

# Prueba de predicciones
y_pred = model.predict(X_test)
print(y_pred)

# Precisión de prueba del modelo
accuracy = accuracy_score(y_test, y_pred)
print("Precisión: ", accuracy)

# Matriz de confusión
confusion_mat = confusion_matrix(y_test, y_pred)
print("Matriz de confusión:")
print(confusion_mat)

C_range = np.logspace(-1, 1, 3)
print(f'La lista de los valores de C es {C_range}')

gamma_range = np.logspace(-1, 1, 3)
print(f'La lista de los valores de gamma es {gamma_range}')