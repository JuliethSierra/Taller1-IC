"""import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Cargar el archivo CSV en un DataFrame
data = pd.read_csv("Prueba1.csv")

# Dividir los datos en características (X) y variable objetivo (y)
print(data.columns)
X = data.drop("Aprobado", axis=1)
y = data["Aprobado"]

# Dividir los datos en conjuntos de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Normalizar los datos
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Crear y entrenar el modelo de regresión logística
model = LogisticRegression()
model.fit(X_train, y_train)

# Realizar predicciones en el conjunto de prueba
y_pred = model.predict(X_test)

# Calcular la precisión del modelo
accuracy = accuracy_score(y_test, y_pred)
print(f'Precisión del modelo: {accuracy:.2f}')
"""


#https://www.aprendemachinelearning.com/regresion-logistica-con-python-paso-a-paso/#more-5394
import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb
#%matplotlib inline

dataframe = pd.read_csv(r"data_finales.csv")
dataframe.head()
dataframe.describe()
#print(dataframe.groupby('Aprobado').size())

#dataframe.drop(['Aprobado'], axis=1)
#dataframe.hist()
#plt.show()

#sb.pairplot(dataframe.dropna(), hue='Aprobado',height=4,vars=["Matematicas","Lenguaje","Ciencias","Ingles"],kind='reg')
print(dataframe)
#plt.show()

sb.heatmap(dataframe.isnull(), cmap='rainbow') #Mostrar campos nulos con un mapa de calor
plt.show()

#crear el modelo
"""X = np.array(dataframe.drop(['Aprobado'], axis=1))
y = np.array(dataframe['Aprobado'])
X.shape

model = linear_model.LogisticRegression()
model.fit(X,y)

predictions = model.predict(X)
#predictions[0:10] imprime los primeros 10 datos de la predicción
print(predictions[0:10])
accuracy=model.score(X,y)
print(f'Precisión del modelo: {accuracy:.2f}')

#Validación del modelo

#dividir los datos
validation_size = 0.30
seed = 7
X_train, X_validation, Y_train, Y_validation = model_selection.train_test_split(X, y, test_size=validation_size, random_state=seed)


name='Logistic Regression'
kfold = model_selection.KFold(n_splits=10, random_state=None)
cv_results = model_selection.cross_val_score(model, X_train, Y_train, cv=kfold, scoring='accuracy')
msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
print(msg)

predictions1 = model.predict(X_validation)
print(accuracy_score(Y_validation, predictions1))

print("\nMatriz de confusión")
print(confusion_matrix(Y_validation, predictions1))


print("\nReporte de clasificación")
print(classification_report(Y_validation, predictions1))


dataframe = pd.read_csv(r"Saber_11__2019-2.csv")
dataframe.head()
dataframe.describe()
print(dataframe.groupby('ESTU_GENERACION-E').size())
dataframe.drop(['ESTU_GENERACION-E'], axis=1)
dataframe.hist()
plt.show()
#sb.pairplot(dataframe.dropna(), hue='ESTU_GENERACION-E',size=10,vars=["DESEMP_SOCIALES_CIUDADANAS","PUNT_INGLES","PERCENTIL_INGLES","DESEMP_INGLES","PUNT_GLOBAL","PERCENTIL_GLOBAL","ESTU_INSE_INDIVIDUAL", "ESTU_NSE_INDIVIDUAL","ESTU_NSE_ESTABLECIMIENTO","ESTU_ESTADOINVESTIGACION"],kind='reg')
#sb.pairplot(dataframe.dropna(), hue='ESTU_GENERACION-E',height=4,vars=["PERCENTIL_GLOBAL","ESTU_INSE_INDIVIDUAL", "ESTU_NSE_INDIVIDUAL","ESTU_NSE_ESTABLECIMIENTO"],kind='reg')
plt.show()


"""