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
        favoritos = datos.get("cultivos_favoritos", [])
        df = pd.DataFrame([datos])

        top_resultado, todos_ordenados = predecir_cultivo_topN(
            modelo, df, le_producto, le_estacion, le_categoria, top_n=3
        )

        cultivos_recomendados = [cultivo for cultivo, _ in top_resultado]
        favoritos_en_top = set(favoritos).intersection(set(cultivos_recomendados))

        if favoritos and not favoritos_en_top:
            for cultivo, prob in todos_ordenados:
                if cultivo in favoritos and cultivo not in cultivos_recomendados:
                    cultivos_recomendados[-1] = cultivo
                    break

        respuesta = []
        for cultivo in cultivos_recomendados:
            prob = next((p for c, p in todos_ordenados if c == cultivo), 0.0)
            respuesta.append({
                "cultivo": cultivo,
                "probabilidad": round(prob * 100, 2)
            })

        return jsonify({"recomendaciones": respuesta})

    except Exception as e:
        return jsonify({"error": str(e)}), 400

port = int(os.environ.get("PORT", 5000))
app.run(host="0.0.0.0", port=port)
