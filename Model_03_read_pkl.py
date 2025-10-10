import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y el scaler desde los archivos .pkl
with open('modelo_regresion_logistica.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# Lista de características en el orden esperado por el modelo
columnas = ['order', 'quotations.prices.estimated', 'quotations.prices.final',
       'quotation', 'mins_to_register', 'mins_to_quote',
       'mins_to_reply', 'mins_to_file', 'mins_to_travel', 'totalMins',
       'marketCode_2H', 'marketCode_2J', 'marketCode_2M', 'marketCode_3N',
       'marketCode_3O', 'marketCode_3P', 'marketCode_HB', 'typeCode_1',
       'typeCode_2', 'typeCode_3', 'typeCode_4', 'typeCode_5', 'typeCode_6',
       'typeCode_7', 'typeCode_8', 'productCode_1', 'productCode_2',
       'productCode_3', 'productCode_4', 'productCode_5', 'departure_day',
       'departure_month']

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
    print("\n*** Mensaje: Es probable que la cotización sea aprobada y concretada en un File :) ***")
else:
    print("\n.La cotización será rechazada :/")

print(f"Probabilidad de aceptación: {probabilidad[0]:.4f}")
