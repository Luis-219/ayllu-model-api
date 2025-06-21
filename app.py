from flask import Flask, request, jsonify
import os
import pandas as pd
import joblib

# === Cargar modelo y transformadores desde el mismo directorio ===
modelo = joblib.load("modelo_rf.pkl")
le_producto = joblib.load("le_productos.pkl")
le_estacion = joblib.load("le_estacion.pkl")
le_categoria = joblib.load("le_categoria.pkl")

with open("features.txt", "r") as f:
    features = f.read().splitlines()

def predecir_cultivo_topN(modelo, df_nuevo, le_producto, le_estacion, le_categoria, top_n=3):
    df_nuevo['ESTACION_SIEMBRA_ENC'] = le_estacion.transform(df_nuevo['ESTACION_SIEMBRA'])
    df_nuevo['CATEGORIA_CULTIVO_ENC'] = le_categoria.transform(df_nuevo['CATEGORIA_CULTIVO'])
    X_nuevo = df_nuevo[features]
    proba = modelo.predict_proba(X_nuevo)[0]
    cultivos = le_producto.inverse_transform(range(len(proba)))
    resultados = list(zip(cultivos, proba))
    resultados_ordenados = sorted(resultados, key=lambda x: x[1], reverse=True)
    return resultados_ordenados[:top_n], resultados_ordenados

app = Flask(__name__)

@app.route("/")
def home():
    return "API de Recomendación de Cultivos activa"

@app.route("/predecir", methods=["POST"])
def predecir():
    try:
        datos = request.get_json()
        df = pd.DataFrame([datos])

        _, todos_ordenados = predecir_cultivo_topN(
            modelo, df, le_producto, le_estacion, le_categoria, top_n=len(le_producto.classes_)
        )

        respuesta = [
            {
                "cultivo": cultivo,
                "probabilidad": round(prob * 100, 2)
            }
            for cultivo, prob in todos_ordenados
        ]

        return jsonify({"recomendaciones": respuesta})

    except Exception as e:
        return jsonify({"error": str(e)}), 400


port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
