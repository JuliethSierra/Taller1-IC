import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Cargar los datos desde el archivo CSV
data = pd.read_csv('data/data_finales.csv')

# Seleccionar las características y la variable objetivo
features = ["ESTU_GENERO","FAMI_ESTRATOVIVIENDA","FAMI_PERSONASHOGAR","FAMI_EDUCACIONPADRE","FAMI_EDUCACIONMADRE","FAMI_TIENEINTERNET","FAMI_TIENESERVICIOTV","FAMI_TIENECOMPUTADOR","FAMI_NUMLIBROS","ESTU_DEDICACIONLECTURADIARIA","ESTU_DEDICACIONINTERNET","ESTU_HORASSEMANATRABAJA","COLE_BILINGUE"]
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
probabilities = model.predict_proba(X_test)[:, 0]

# Calcular el promedio de las probabilidades
average_probability = probabilities.mean()

average_percentage = average_probability * 100
average_percentage_no_generacione = model.predict_proba(X_test)[:, 1].mean() * 100

print(f'Probabilidad de que un estudiante sea Generación E es del {average_percentage:.2f}%')
print(f"Probabilidad de que un estudiante no sea Generación E es del: {average_percentage_no_generacione:.2f}%")

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)*100
print(f'Precisión del modelo: {accuracy:.2f}%')

# Calcular la matriz de confusión
confusion = confusion_matrix(y_test, y_pred)

# Crear una gráfica de matriz de confusión
plt.figure(figsize=(8, 6))
sns.heatmap(confusion, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.xlabel('Etiqueta Predicha')
plt.ylabel('Etiqueta Real')
plt.title('Matriz de Confusión')
plt.show()