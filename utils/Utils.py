import pandas as pd
import numpy as np
from sklearn import linear_model
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sb

dataframe = pd.read_csv(r"data\data_finales.csv")
#dataframe.head()
#dataframe.describe()
#print(dataframe.groupby('Aprobado').size())

#dataframe.drop(['Aprobado'], axis=1)
#dataframe.hist()
#plt.show()

#sb.pairplot(dataframe.dropna(), hue='Aprobado',height=4,vars=["Matematicas","Lenguaje","Ciencias","Ingles"],kind='reg')
#print(dataframe)
#plt.show()

#sb.heatmap(dataframe.isnull(), cmap='rainbow') #Mostrar campos nulos con un mapa de calor
#plt.show()

dataframe.drop(['ESTU_GENERACION-E'], axis=1).hist() #visualización de datos
plt.show()


