from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import os
from datetime import datetime

app = Flask(__name__)

# --- Carga de Modelos ---
modelo_sys = joblib.load("modelo_sys.pkl") if os.path.exists("modelo_sys.pkl") else None
modelo_dia = joblib.load("modelo_dia.pkl") if os.path.exists("modelo_dia.pkl") else None

ultima_estimacion = {
    "sys": "---", "dia": "---", "spo2": "---",
    "hr": "---", "nivel": "---", "timestamp": "---",
    "hr_sensor": "---", "spo2_sensor": "---", "ir": "---", "red": "---"
}

# --- Clasificación del estado ---
def clasificar_nivel_presion(pas, pad):
    try:
        pas = float(pas)
        pad = float(pad)
        if pas > 180 or pad > 120: return "HT Crisis"
        elif pas >= 140 or pad >= 90: return "HT2"
        elif (130 <= pas <= 139) or (80 <= pad <= 89): return "HT1"
        elif 120 <= pas <= 129 and pad < 80: return "Elevada"
        elif pas < 120 and pad < 80: return "Normal"
        else: return "Revisar"
    except: return "---"

@app.route("/")
def home():
    return render_template("index.html", estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def recibir_datos():
    global ultima_estimacion
    data = request.get_json()
    try:
        hr = int(data.get("hr", -1))
        spo2 = float(data.get("spo2", -1))
        ir = int(data.get("ir", -1))
        red = int(data.get("red", -1))

        # Guardar datos del sensor
        ultima_estimacion["hr_sensor"] = hr
        ultima_estimacion["spo2_sensor"] = spo2
        ultima_estimacion["ir"] = ir
        ultima_estimacion["red"] = red

        # Predicción con ML
        if modelo_sys and modelo_dia:
            entrada = pd.DataFrame([[hr, spo2]], columns=["hr", "spo2"])
            sys = round(float(modelo_sys.predict(entrada)[0]), 2)
            dia = round(float(modelo_dia.predict(entrada)[0]), 2)
            nivel = clasificar_nivel_presion(sys, dia)
            ultima_estimacion.update({
                "sys": sys, "dia": dia, "spo2": spo2,
                "hr": hr, "nivel": nivel,
                "timestamp": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
            })
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route("/api/ultima_estimacion", methods=["GET"])
def get_ultima():
    return jsonify(ultima_estimacion)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 10000)), debug=True)
