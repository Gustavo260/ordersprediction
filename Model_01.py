# Importar las bibliotecas necesarias
import pandas as pd
import numpy as np

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
data = pd.read_csv('dataset_orders.csv')

# Se limpia variables de los puntos
data = data.rename(columns={
    'quotations.prices.estimated': 'quotations_prices_estimated',
    'quotations.prices.final': 'quotations_prices_final'
})

# Exploración inicial de los datos
print("Primeras filas del conjunto de datos:")
print(data.head())
print("\nDescripción estadística:")
print(data.describe())
print("\nDistribución de la variable objetivo:")
print(data['file'].value_counts())

# Verificar valores nulos
print("\nValores nulos en cada columna:")
print(data.isnull().sum())

# MANEJO DE VALORES NaN - CORRECCIÓN
# Eliminar filas con valores NaN en las características
data_clean = data.dropna()
print(f"\nDatos después de eliminar NaN: {data_clean.shape}")

# Separar características y variable objetivo
X = data_clean.drop('file', axis=1)
y = data_clean['file']

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

# Evaluación del modelo
print("\nReporte de clasificación para Regresión Logística:")
print(classification_report(y_test, y_pred))

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Matriz de Confusión - Regresión Logística')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Calcular y mostrar el ROC AUC Score
roc_auc = roc_auc_score(y_test, y_prob)
print("ROC AUC Score para Regresión Logística:", roc_auc)

# Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label='Regresión Logística (área = %0.3f)' % roc_auc)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Regresión Logística')
plt.legend()
plt.show()

# Curva Precision-Recall
precision, recall, thresholds_pr = precision_recall_curve(y_test, y_prob)
plt.figure(figsize=(6,4))
plt.plot(recall, precision, label='Regresión Logística')
plt.xlabel('Recall')
plt.ylabel('Precisión')
plt.title('Curva Precision-Recall - Regresión Logística')
plt.legend()
plt.show()


# Entrenar un modelo de Bosque Aleatorio
rf = RandomForestClassifier(n_estimators=100, class_weight='balanced', random_state=42)
rf.fit(X_train_scaled, y_train)

# Realizar predicciones con el modelo de Bosque Aleatorio
y_pred_rf = rf.predict(X_test_scaled)
y_prob_rf = rf.predict_proba(X_test_scaled)[:, 1]

# Evaluación del modelo de Bosque Aleatorio
print("\nReporte de clasificación para Bosque Aleatorio:")
print(classification_report(y_test, y_pred_rf))

# Matriz de confusión para Bosque Aleatorio
cm_rf = confusion_matrix(y_test, y_pred_rf)
plt.figure(figsize=(6,4))
sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens')
plt.title('Matriz de Confusión - Bosque Aleatorio')
plt.xlabel('Predicción')
plt.ylabel('Real')
plt.show()

# Calcular y mostrar el ROC AUC Score para Bosque Aleatorio
roc_auc_rf = roc_auc_score(y_test, y_prob_rf)
print("ROC AUC Score para Bosque Aleatorio:", roc_auc_rf)

# Curva ROC para Bosque Aleatorio
fpr_rf, tpr_rf, thresholds_rf = roc_curve(y_test, y_prob_rf)
plt.figure(figsize=(6,4))
plt.plot(fpr_rf, tpr_rf, label='Bosque Aleatorio (área = %0.3f)' % roc_auc_rf)
plt.plot([0,1], [0,1], 'r--')
plt.xlabel('Tasa de Falsos Positivos')
plt.ylabel('Tasa de Verdaderos Positivos')
plt.title('Curva ROC - Bosque Aleatorio')
plt.legend()
plt.show()