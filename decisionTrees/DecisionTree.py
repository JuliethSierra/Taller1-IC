import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# Cargar los datos desde el archivo CSV
data = pd.read_csv('data/data_finales.csv')

# Seleccionar las características y la variable objetivo
features = ['ESTU_GENERO', 'FAMI_ESTRATOVIVIENDA', 'FAMI_TIENEINTERNET', 'FAMI_TIENECOMPUTADOR', 'COLE_BILINGUE']
target = 'ESTU_GENERACION-E'

X = data[features]
y = data[target]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Crear el modelo de árbol de decisión
model = DecisionTreeClassifier(random_state=42)

# Entrenar el modelo
model.fit(X_train, y_train)

# Calcular las probabilidades de que la variable objetivo sea 1 en el conjunto de prueba
probabilities = model.predict_proba(X_test)[:, 1]

# Calcular el promedio de las probabilidades
average_probability = probabilities.mean()

average_percentage = average_probability * 100

print(f'Probabilidad de que un estudiante sea Generación E es del {average_percentage:.2f}%')
