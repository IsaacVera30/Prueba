from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime

app = Flask(__name__, template_folder='templates')
CORS(app)

# Cargar modelos entrenados
modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

# CSV para entrenamiento futuro
CSV_PATH = "registro_sensor_entrenamiento.csv"

# Variables para control de registros por autorización
registros_autorizados = []
MAX_REGISTROS = 15
valores_referencia = {}

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/health")
def health():
    return "OK", 200

@app.route("/api/presion", methods=["POST"])
def recibir_datos():
    data = request.get_json()

    hr_crudo = data.get("hr")
    ir = data.get("ir")
    red = data.get("red")
    id_paciente = data.get("id_paciente", 1)
    
    # --- Cálculos sencillos (modo robusto simulado) ---
    hr_promedio_sensor = np.clip(hr_crudo, 60, 100) + np.random.normal(0, 1)
    spo2_sensor = 100 - (abs(ir - red) / max(ir + red, 1)) * 100

    # Valores del modelo
    entrada_ml = np.array([[hr_promedio_sensor, spo2_sensor]])
    sys_ml = modelo_sys.predict(entrada_ml)[0]
    dia_ml = modelo_dia.predict(entrada_ml)[0]

    hr_ml = hr_promedio_sensor + np.random.normal(0, 0.5)
    spo2_ml = spo2_sensor + np.random.normal(0, 0.3)

    # Estado estimado
    if sys_ml >= 180 or dia_ml >= 120:
        estado = "HT Crisis"
    elif sys_ml >= 160 or dia_ml >= 100:
        estado = "HT2"
    elif sys_ml >= 140 or dia_ml >= 90:
        estado = "HT1"
    elif sys_ml >= 130 or dia_ml >= 85:
        estado = "Elevada"
    else:
        estado = "Normal"

    # Guardar en CSV si hay valores de referencia
    if valores_referencia:
        for _ in range(MAX_REGISTROS):
            registros_autorizados.append({
                "hr_crudo": hr_crudo,
                "hr_promedio_sensor": hr_promedio_sensor,
                "spo2_sensor": spo2_sensor,
                "ir": ir,
                "red": red,
                "hr_referencia": valores_referencia["hr"],
                "pas_referencia": valores_referencia["sys"],
                "pad_referencia": valores_referencia["dia"]
            })

        # Guardar y limpiar buffer
        df_actual = pd.DataFrame(registros_autorizados)
        if os.path.exists(CSV_PATH):
            df_actual.to_csv(CSV_PATH, mode='a', index=False, header=False)
        else:
            df_actual.to_csv(CSV_PATH, index=False)
        registros_autorizados.clear()
        valores_referencia.clear()

    # Retorno al ESP32
    return jsonify({
        "sys": round(sys_ml, 1),
        "dia": round(dia_ml, 1),
        "hr": round(hr_ml, 1),
        "spo2": round(spo2_ml, 1),
        "nivel": estado
    })

@app.route("/api/autorizar", methods=["POST"])
def autorizar_guardado():
    data = request.get_json()
    valores_referencia["sys"] = data.get("pas_referencia")
    valores_referencia["dia"] = data.get("pad_referencia")
    valores_referencia["hr"] = data.get("hr_referencia")
    return jsonify({"mensaje": "Autorizado para guardar próximos registros."})

@app.route("/api/registro_count", methods=["GET"])
def obtener_cantidad_registros():
    return jsonify({"restantes": MAX_REGISTROS})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
