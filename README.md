# ordersprediction

Proyecto basado en pedidos (solicitud de cotizaciones), con la intención de predecir si sería concretada en venta o no.



Descripción general

--------------------

Esta aplicación predice la probabilidad de que una cotización sea aprobada o concretada en una venta (variable 'file' = 1).

El modelo fue desarrollado con FastAPI y entrenado con un modelo de regresión logística usando un dataset de cotizaciones históricas.



Variable objetivo

--------------------

La variable 'file' representa la venta o aprobación de la cotización, indicando que el pedido fue concretado satisfactoriamente cuando es 1/True.



Estructura del proyecto

------------------------

\- Model\_04\_FASTAPI.py         → API principal en FastAPI

\- modelo\_regresion\_logistica.pkl → Modelo entrenado

\- scaler.pkl                     → Escalador usado para normalizar los datos

\- requirements.txt               → Dependencias necesarias

\- runtime.txt                    → Versión de Python

\- dataset\_orders.csv             → Dataset base para el modelo



Descripción de variables

------------------------

\- order: Identificador del pedido o cotización.

\- quotations.prices.estimated: Precio estimado inicial de la cotización.

\- quotations.prices.final: Precio final cotizado tras ajustes.

\- quotation: Indica si existe una cotización (1) o no (0).

\- file: Variable objetivo; indica si la cotización fue concretada (1 = venta aprobada).

\- mins\_to\_register: Minutos transcurridos hasta registrar la cotización.

\- mins\_to\_quote: Minutos empleados para generar la cotización.

\- mins\_to\_reply: Minutos hasta recibir una respuesta del cliente.

\- mins\_to\_file: Minutos entre cotización y archivo o cierre del pedido.

\- mins\_to\_travel: Tiempo estimado de traslado asociado al pedido.

\- totalMins: Suma total de tiempos operativos involucrados.

\- marketCode\_2H a marketCode\_HB: Variables dummy que representan distintos mercados o canales.

\- typeCode\_1 a typeCode\_8: Tipos de producto o servicio cotizado, codificados como variables binarias.

\- productCode\_1 a productCode\_5: Identificadores de familias de producto específicas.

\- departure\_day: Día del mes en que inicia o sale el servicio.

\- departure\_month: Mes de salida o ejecución del pedido.



Endpoint principal

------------------

POST /prediccion/

Devuelve la probabilidad de que una cotización sea aceptada.



Link de acceso a la app: https://ordersprediction-cc.onrender.com/docs



Ejemplo de entrada (Aceptado):

{
  "order": 4,
  "quotations_prices_estimated": 778,
  "quotations_prices_final": 777,
  "quotation": 4,
  "mins_to_register": 0,
  "mins_to_quote": 23,
  "mins_to_reply": 23,
  "mins_to_file": 5,
  "mins_to_travel": 0,
  "totalMins": 0,
  "marketCode_2H": 0,
  "marketCode_2J": 0,
  "marketCode_2M": 0,
  "marketCode_3N": 0,
  "marketCode_3O": 0,
  "marketCode_3P": 0,
  "marketCode_HB": 0,
  "typeCode_1": 0,
  "typeCode_2": 0,
  "typeCode_3": 0,
  "typeCode_4": 0,
  "typeCode_5": 0,
  "typeCode_6": 0,
  "typeCode_7": 0,
  "typeCode_8": 0,
  "productCode_1": 0,
  "productCode_2": 0,
  "productCode_3": 0,
  "productCode_4": 0,
  "productCode_5": 0,
  "departure_day": 6,
  "departure_month": 12
}


Ejemplo de salida:

{
  "Resultado": "Aprobado",
  "Probabilidad de que la cotización SI sea aceptada": "65.98%",
  "Probabilidad de que la cotización NO sea aceptada": "34.02%"
}



Ejemplo de entrada (Rechazado) (Bajamos el price_final a 700):

{
  "order": 4,
  "quotations_prices_estimated": 778,
  "quotations_prices_final": 700,
  "quotation": 4,
  "mins_to_register": 0,
  "mins_to_quote": 23,
  "mins_to_reply": 23,
  "mins_to_file": 5,
  "mins_to_travel": 0,
  "totalMins": 0,
  "marketCode_2H": 0,
  "marketCode_2J": 0,
  "marketCode_2M": 0,
  "marketCode_3N": 0,
  "marketCode_3O": 0,
  "marketCode_3P": 0,
  "marketCode_HB": 0,
  "typeCode_1": 0,
  "typeCode_2": 0,
  "typeCode_3": 0,
  "typeCode_4": 0,
  "typeCode_5": 0,
  "typeCode_6": 0,
  "typeCode_7": 0,
  "typeCode_8": 0,
  "productCode_1": 0,
  "productCode_2": 0,
  "productCode_3": 0,
  "productCode_4": 0,
  "productCode_5": 0,
  "departure_day": 6,
  "departure_month": 12
}


Ejemplo de salida:

{
  "Resultado": "Rechazado",
  "Probabilidad de que la cotización SI sea aceptada": "49.05%",
  "Probabilidad de que la cotización NO sea aceptada": "50.95%"
}

Notas finales

\- Si / muestra 404, es normal (la ruta principal es /prediccion/ o /docs).

\- Si cambias el código o modelo, usa 'Manual Deploy → Deploy latest commit' en Render.

\- La aplicación fue diseñada con fines académicos y de demostración.

