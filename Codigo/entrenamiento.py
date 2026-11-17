import os
import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


raiz_proyecto = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ruta_datos = os.path.join(
    raiz_proyecto, "Data", "DataSet Viviendas_entrenamiento+test.xlsx"
)
ruta_modelos = os.path.join(raiz_proyecto, "Modelos")


df = pd.read_excel(ruta_datos)


col_categoricas = df.select_dtypes(include=["object"]).columns.tolist()
codificador = OneHotEncoder(drop="first", sparse_output=False)
categoricos_codificados = codificador.fit_transform(df[col_categoricas])

columnas = codificador.get_feature_names_out(col_categoricas)
df_codificado = pd.DataFrame(categoricos_codificados, columns=columnas)


df_model = pd.concat(
    [df[["Superficie", "Antiguedad", "VPO", "Precio"]], df_codificado], axis=1
)
X = df_model.drop("Precio", axis=1)
y = df_model["Precio"]


X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
model = LinearRegression()
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(f"MSE: {mse:.2f} | RMSE: {rmse:.2f}")


joblib.dump(codificador, os.path.join(ruta_modelos, "onehotencoder.pkl"))
joblib.dump(model, os.path.join(ruta_modelos, "linear_regression_model.pkl"))
print("Archivos .pkl guardados correctamente en la carpeta Modelos.")