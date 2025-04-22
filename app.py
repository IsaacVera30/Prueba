from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import os

app = Flask(__name__)
modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

# Variables globales
autorizado = False
ultima_prediccion = {"sys": None, "dia": None}

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", autorizado=autorizado,
                           sys=ultima_prediccion["sys"],
                           dia=ultima_prediccion["dia"])

@app.route("/api/datos", methods=["POST"])
def recibir_datos():
    global ultima_prediccion

    data = request.get_json()
    hr = int(data.get("hr", 0))
    spo2 = int(data.get("spo2", 0))
    ir = int(data.get("ir", 0))
    red = int(data.get("red", 0))

    # Predecir presión con ML
    entrada = pd.DataFrame([[hr, spo2]], columns=["hr", "spo2"])
    sys = modelo_sys.predict(entrada)[0]
    dia = modelo_dia.predict(entrada)[0]
    ultima_prediccion = {"sys": round(sys, 2), "dia": round(dia, 2)}

    # Si está autorizado, guardar en CSV
    if autorizado:
        with open("registro_sensor_entrenamiento.csv", "a") as file:
            file.write(f"{hr},{spo2},{ir},{red},{sys:.2f},{dia:.2f}\n")

    return jsonify({
        "mensaje": "✅ Datos procesados",
        "sys": round(sys, 2),
        "dia": round(dia, 2)
    }), 200

@app.route("/api/presion", methods=["POST"])
def autorizar_registro():
    global autorizado
    autorizado = True
    return render_template("index.html", autorizado=autorizado,
                           sys=ultima_prediccion["sys"],
                           dia=ultima_prediccion["dia"])

@app.route("/api/detener", methods=["POST"])
def detener_registro():
    global autorizado
    autorizado = False
    return render_template("index.html", autorizado=autorizado,
                           sys=ultima_prediccion["sys"],
                           dia=ultima_prediccion["dia"])

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
