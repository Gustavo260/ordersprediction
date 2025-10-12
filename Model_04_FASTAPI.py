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
columnas = ['order', 'quotations.prices.estimated', 'quotations.prices.final',
       'quotation', 'mins_to_register', 'mins_to_quote',
       'mins_to_reply', 'mins_to_file', 'mins_to_travel', 'totalMins',
       'marketCode_2H', 'marketCode_2J', 'marketCode_2M', 'marketCode_3N',
       'marketCode_3O', 'marketCode_3P', 'marketCode_HB', 'typeCode_1',
       'typeCode_2', 'typeCode_3', 'typeCode_4', 'typeCode_5', 'typeCode_6',
       'typeCode_7', 'typeCode_8', 'productCode_1', 'productCode_2',
       'productCode_3', 'productCode_4', 'productCode_5', 'departure_day',
       'departure_month']
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
