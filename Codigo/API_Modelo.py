import os
from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI()


raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_modelos = os.path.join(raiz_proyecto, "Modelos")


codificador = joblib.load(os.path.join(ruta_modelos, "onehotencoder.pkl"))
model = joblib.load(os.path.join(ruta_modelos, "linear_regression_model.pkl"))


def obtener_prediccion(
    superficie, antiguedad, vpo, calefaccion, habitaciones, ascensor, garage
):
  col_categoricas = ["Calefaccion", "Habitaciones", "Ascensor", "Garaje"]

  pred_data = pd.DataFrame({
      "Superficie": [superficie],
      "Antiguedad": [antiguedad],
      "VPO": [vpo],
      "Calefaccion": [calefaccion],
      "Habitaciones": [habitaciones],
      "Ascensor": [ascensor],
      "Garaje": [garage],
  })

  nuevo_codificado = codificador.transform(pred_data[col_categoricas])
  df_nuevo_codificado = pd.DataFrame(
      nuevo_codificado,
      columns=codificador.get_feature_names_out(col_categoricas),
  )

  pred_data_final = pd.concat(
      [pred_data[["Superficie", "Antiguedad", "VPO"]], df_nuevo_codificado],
      axis=1,
  )

  prediccion = model.predict(pred_data_final)
  return prediccion.tolist()


@app.get("/prediccion/")
async def prediccion(
    superficie: float,
    antiguedad: float,
    vpo: int,
    calefaccion: str,
    habitaciones: str,
    ascensor: str,
    garage: str,
):
  return obtener_prediccion(
      superficie, antiguedad, vpo, calefaccion, habitaciones, ascensor, garage
  )