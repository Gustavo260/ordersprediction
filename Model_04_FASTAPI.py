from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse, RedirectResponse
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

app = FastAPI(title="Determinar si una cotización pueda o no ser aceptada")

# Redirección a la documentación
#@app.get("/")
#def redirect_to_docs():
    #return RedirectResponse(url="/docs")

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

@app.post("/prediccion/")
async def predecir_cotizacion(transaccion: Transaccion):
    try:
        # DataFrame con el orden exacto de columnas
        df = pd.DataFrame([transaccion.dict()], columns=columnas)

        # Escalado
        X_scaled = scaler.transform(df)

        # Predicción y probabilidad clase positiva (SI aceptada)
        y_pred = modelo.predict(X_scaled)[0]
        p_si = float(modelo.predict_proba(X_scaled)[:, 1][0]) * 100
        p_no = 100 - p_si

        # Formateo a 2 decimales COMO STRING para evitar enteros tipo 0/1
        p_si_str = f"{p_si:.1f}%"
        p_no_str = f"{p_no:.1f}%"

        resultado = {
            "Resultado": "Se acepta" if bool(y_pred) else "No se acepta",
            "Probabilidad de que la cotización SI sea aceptada": p_si_str,
            "Probabilidad de que la cotización NO sea aceptada": p_no_str
        }
        return JSONResponse(content=resultado)

    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))