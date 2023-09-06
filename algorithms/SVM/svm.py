import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

data = pd.read_csv('data/data_finales.csv')

X = data.drop('ESTU_GENERACION-E', axis=1)  # Características
y = data['ESTU_GENERACION-E']  # Variable objetivo

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=1000, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

svm_model = SVC(kernel='linear', probability=True)
calibrated_svm = CalibratedClassifierCV(svm_model)
calibrated_svm.fit(X_train, y_train)

# Calcula las probabilidades
y_probabilities = calibrated_svm.predict_proba(X_test)

# Calcula la probabilidad promedio de ser Generación E en todo el conjunto de prueba
probabilidad_promedio = y_probabilities[:, 0].mean()

average_percentage = probabilidad_promedio * 100

print(f"Probabilidad de que un estudiante sea Generación E: {average_percentage:.2f}%")
print(f"Probabilidad de que un estudiante no sea Generación E: {y_probabilities[:, 1].mean()*100:.2f}%")

# Realizar predicciones en el conjunto de prueba
y_pred = calibrated_svm.predict(X_test)

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