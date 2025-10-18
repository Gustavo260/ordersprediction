from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
from fastapi.responses import RedirectResponse
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
columnas = [
    'order', 'quotations_prices_estimated', 'quotations_prices_final',
    'quotation', 'mins_to_register', 'mins_to_quote', 'mins_to_reply',
    'mins_to_file', 'mins_to_travel', 'totalMins', 'marketCode_2H',
    'marketCode_2J', 'marketCode_2M', 'marketCode_3N', 'marketCode_3O',
    'marketCode_3P', 'marketCode_HB', 'typeCode_1', 'typeCode_2',
    'typeCode_3', 'typeCode_4', 'typeCode_5', 'typeCode_6', 'typeCode_7',
    'typeCode_8', 'productCode_1', 'productCode_2', 'productCode_3',
    'productCode_4', 'productCode_5', 'departure_day', 'departure_month'
]

# Crear la aplicación FastAPI
app = FastAPI(title="Determinar si una cotización pueda o no ser aceptada")

# Definir el modelo de datos de entrada utilizando Pydantic
class Transaccion(BaseModel):
    order: float
    quotations_prices_estimated: float
    quotations_prices_final: float
    quotation: float
    mins_to_register: float
    mins_to_quote: float
    mins_to_reply: float
    mins_to_file: float
    mins_to_travel: float
    totalMins: float
    marketCode_2H: float
    marketCode_2J: float
    marketCode_2M: float
    marketCode_3N: float
    marketCode_3O: float
    marketCode_3P: float
    marketCode_HB: float
    typeCode_1: float
    typeCode_2: float
    typeCode_3: float
    typeCode_4: float
    typeCode_5: float
    typeCode_6: float
    typeCode_7: float
    typeCode_8: float
    productCode_1: float
    productCode_2: float
    productCode_3: float
    productCode_4: float
    productCode_5: float
    departure_day: float
    departure_month: float
    
@app.get("/")
def redirect_to_docs():
    return RedirectResponse(url="/docs")

# Definir el endpoint para predicción
@app.post("/prediccion/")
async def predecir_cotizacion(transaccion: Transaccion):
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
            "Probabilidad de que la cotización SI sea aceptada": bool(prediccion[0]),
            "Probabilidad de que la cotización NO sea aceptada": float(probabilidad[0])
        }

        # Retornar el resultado directamente en formato JSON
        return JSONResponse(content=resultado)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))