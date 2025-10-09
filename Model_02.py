# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np
import pickle

# Para visualización de datos
import matplotlib.pyplot as plt
import seaborn as sns

# Para preprocesamiento y construcción del modelo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve

# Cargar el conjunto de datos
data = pd.read_csv('E:/Bismack-D/UNIVERSIDAD Adolfo Ibañez/07- Cloud Computing 1/Clase4/creditcard.csv')

# Separar características y variable objetivo
X = data.drop('Class', axis=1)
y = data['Class']

# Dividir los datos en conjuntos de entrenamiento y prueba con estratificación debido al desbalance
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Escalamiento de características
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Entrenar un modelo de Regresión Logística con balanceo de clases
lr = LogisticRegression(class_weight='balanced', random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)

# Realizar predicciones
y_pred = lr.predict(X_test_scaled)
y_prob = lr.predict_proba(X_test_scaled)[:, 1]


# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)


# Calcular y mostrar el ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)


# Curva Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)



# Entrenar un modelo de Bosque Aleatorio
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)

# Realizar predicciones con el modelo de Bosque Aleatorio
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Matriz de confusión para Bosque Aleatorio
cm_rf = confusion_matrix(y_test, y_pred_rf)

# Calcular y mostrar el ROC AUC Score para Bosque Aleatorio
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)

# Curva ROC para Bosque Aleatorio
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)


with open('modelo_regresion_logistica.pkl', 'wb') as archivo_salida:
    pickle.dump(lr, archivo_salida)

# Guardar el modelo de Bosque Aleatorio en un archivo .pkl
with open('modelo_bosque_aleatorio.pkl', 'wb') as archivo_salida:
    pickle.dump(rf, archivo_salida)

# Si también deseas guardar el scaler
with open('scaler.pkl', 'wb') as archivo_salida:
    pickle.dump(scaler, archivo_salida)