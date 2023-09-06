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

print("Start")
data = pd.read_csv('data/data_finales.csv')
print("Readed data")

X = data.drop('ESTU_GENERACION-E', axis=1)  # Características
y = data['ESTU_GENERACION-E']  # Variable objetivo

print("Train")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Finish train")

print("Scaler")
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
print("Finish scaler")

print("Model")
svm_model = SVC(kernel='linear', probability=True)
print("Calibrate")
calibrated_svm = CalibratedClassifierCV(svm_model)
print("Finish calibrate")
calibrated_svm.fit(X_train, y_train)
print("Finish Model")

# Calcula las probabilidades
y_probabilities = calibrated_svm.predict_proba(X_test)

# Calcula la probabilidad promedio de ser Generación E en todo el conjunto de prueba
probabilidad_promedio = y_probabilities[:, 1].mean()

print(f"Probabilidad promedio de ser Generación E: {probabilidad_promedio:.2f}")