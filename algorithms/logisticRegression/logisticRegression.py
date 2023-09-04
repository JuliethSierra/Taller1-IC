import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar el archivo CSV en un DataFrame
data = pd.read_csv("data\data_finales.csv")

# Dividir los datos en características (X) y variable objetivo (y)
print(data.columns)
X = data.drop("ESTU_GENERACION-E", axis=1)
y = data["ESTU_GENERACION-E"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular las probabilidades en el conjunto de prueba
probabilities = model.predict_proba(X_test)

# Calcular el porcentaje total de las probabilidades para cada clase
total_percentages = probabilities.sum(axis=0) / len(y_test) * 100

# Imprimir los porcentajes totales para cada clase
print(f"Probabilidad de que un estudiante sea Generación E es del: {total_percentages[0]:.2f}%")
print(f"Probabilidad de que un estudiante no sea Generación E es del: {total_percentages[1]:.2f}%")

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')