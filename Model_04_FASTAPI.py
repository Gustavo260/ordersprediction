from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import numpy as np
import pandas as pd

# Cargar el modelo y el scaler desde los archivos .pkl
with open('modelo_regresion_logistica.pkl', 'rb') as archivo_modelo:
    modelo = pickle.load(archivo_modelo)

with open('scaler.pkl', 'rb') as archivo_scaler:
    scaler = pickle.load(archivo_scaler)

# Definir las características esperadas
columnas = ['Time', 'V1', 'V2', 'V3', 'V4', 'V5', 'V6', 'V7', 'V8', 'V9', 'V10',
            'V11', 'V12', 'V13', 'V14', 'V15', 'V16', 'V17', 'V18', 'V19', 'V20',
            'V21', 'V22', 'V23', 'V24', 'V25', 'V26', 'V27', 'V28', 'Amount']

# Crear la aplicación FastAPI
app = FastAPI(title="Detección de Fraude en Transacciones")

# Definir el modelo de datos de entrada utilizando Pydantic
class Transaccion(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Definir el endpoint para predicción
@app.post("/prediccion/")
async def predecir_fraude(transaccion: Transaccion):
    try:
        # Convertir la entrada en un DataFrame
        datos_entrada = pd.DataFrame([transaccion.dict()], columns=columnas)
        
        # Escalar las características
        datos_entrada_scaled = scaler.transform(datos_entrada)
        
        # Realizar la predicción
        prediccion = modelo.predict(datos_entrada_scaled)
        probabilidad = modelo.predict_proba(datos_entrada_scaled)[:, 1]
        
        # Construir la respuesta
        resultado = {
            "EsFraude": bool(prediccion[0]),
            "ProbabilidadFraude": float(probabilidad[0])
        }
        
        return resultado
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
