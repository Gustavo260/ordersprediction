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



Ejemplo de entrada:

{

&nbsp; "order": 4,

&nbsp; "quotations\_prices\_estimated": 778,

&nbsp; "quotations\_prices\_final": 777,

&nbsp; "quotation": 4,

&nbsp; "mins\_to\_register": 0,

&nbsp; "mins\_to\_quote": 23,

&nbsp; "mins\_to\_reply": 23,

&nbsp; "mins\_to\_file": 5,

&nbsp; "mins\_to\_travel": 0,

&nbsp; "totalMins": 0,

&nbsp; "marketCode\_2H": 0,

&nbsp; "marketCode\_2J": 0,

&nbsp; "marketCode\_2M": 0,

&nbsp; "marketCode\_3N": 0,

&nbsp; "marketCode\_3O": 0,

&nbsp; "marketCode\_3P": 0,

&nbsp; "marketCode\_HB": 0,

&nbsp; "typeCode\_1": 0,

&nbsp; "typeCode\_2": 0,

&nbsp; "typeCode\_3": 0,

&nbsp; "typeCode\_4": 0,

&nbsp; "typeCode\_5": 0,

&nbsp; "typeCode\_6": 0,

&nbsp; "typeCode\_7": 0,

&nbsp; "typeCode\_8": 0,

&nbsp; "productCode\_1": 0,

&nbsp; "productCode\_2": 0,

&nbsp; "productCode\_3": 0,

&nbsp; "productCode\_4": 0,

&nbsp; "productCode\_5": 0,

&nbsp; "departure\_day": 6,

&nbsp; "departure\_month": 12

}



Ejemplo de salida:

{

&nbsp; "Resultado": "Se acepta",

&nbsp; "Probabilidad de que la cotización SI sea aceptada": "66.0%",

&nbsp; "Probabilidad de que la cotización NO sea aceptada": "34.0%"

}



Notas finales

\- Si / muestra 404, es normal (la ruta principal es /prediccion/ o /docs).

\- Si cambias el código o modelo, usa 'Manual Deploy → Deploy latest commit' en Render.

\- La aplicación fue diseñada con fines académicos y de demostración.

