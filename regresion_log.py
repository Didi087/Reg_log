# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

###importar librerías
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

###cargamos base
mydataset = pd.read_csv('data1.csv')
x = mydataset[['PhoneService','tenure']].values
y = mydataset[['Churn']].values

###obtenemos conjunto de entrenamiento y pruebas

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.25,random_state=0)


###estandarización de escalas
from sklearn.preprocessing import StandardScaler
sc =StandardScaler()
x_train = sc.fit_transform(x_train)
x_test = sc.transform(x_test)

###Regresión logistica

from sklearn.linear_model import LogisticRegression
clasificador = LogisticRegression(random_state=0)
clasificador.fit(x_train,y_train)


###Predicción

y_pred = clasificador.predict(x_test)


###matriz de confusión

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)

score = clasificador.score(x_test,y_test)

###Gráfica de pruebas

##from matplotlib.colors import ListedColormap
#x_set, y_set = x_test, y_test
#X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() - 1, stop = x_set[:, 0].max() + 1, step = 0.01),
#                     np.arange(start = x_set[:, 1].min() - 1, stop = x_set[:, 1].max() + 1, step = 0.01))
#plt.contourf(X1, X2, clasificador.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
#            alpha = 0.75, cmap = ListedColormap(('red', 'blue')))
#plt.xlim(X1.min(), X1.max())
#plt.ylim(X2.min(), X2.max())
#for i, j in enumerate(np.unique(y_set)):
 #   plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1],
 #               c = ListedColormap(('red', 'blue'))(i), label = j)
#plt.title('Regresion Logistica Prueba')
#plt.xlabel('tenure')
#plt.ylabel('Churn')
#plt.legend()
#plt.show()


###árbol decisión (minimizar entropía (insertidumbre))

from sklearn.tree import DecisionTreeClassifier
clasificador2 = DecisionTreeClassifier(criterion = 'entropy', random_state=0)
clasificador2.fit(x_train,y_train)

y_pred_tree = clasificador2.predict(x_test)

cm2 =confusion_matrix(y_test, y_pred_tree)

score2 = clasificador2.score(x_test,y_test)
















