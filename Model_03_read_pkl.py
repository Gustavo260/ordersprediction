import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y el scaler desde los archivos .pkl
with open('modelo_regresion_logistica.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# Lista de características en el orden esperado por el modelo
columnas = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Crear una lista para almacenar los valores ingresados por el usuario
valores_usuario = []

print("Por favor, ingrese los siguientes valores:")
print("(Nota: Ingrese los valores numéricos correspondientes a cada característica.)\n")

for columna in columnas:
    while True:
        try:
            valor = float(input(f"{columna}: "))
            valores_usuario.append(valor)
            break
        except ValueError:
            print("Entrada inválida. Por favor, ingrese un número válido.")

# Convertir los valores a un DataFrame con las columnas correctas
nueva_muestra = pd.DataFrame([valores_usuario], columns=columnas)

# Escalar las características utilizando el scaler cargado
nueva_muestra_scaled = scaler.transform(nueva_muestra)

# Realizar la predicción con el modelo cargado
prediccion = modelo.predict(nueva_muestra_scaled)
probabilidad = modelo.predict_proba(nueva_muestra_scaled)[:, 1]

# Mostrar el resultado al usuario
if prediccion[0] == 1:
    print("\n*** Alerta: La transacción es FRAUDULENTA ***")
else:
    print("\nLa transacción es legítima.")

print(f"Probabilidad de fraude: {probabilidad[0]:.4f}")
