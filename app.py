
from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np
import mysql.connector
from datetime import datetime

app = Flask(__name__)

modelo_sys = joblib.load("modelo_sys.pkl")
modelo_dia = joblib.load("modelo_dia.pkl")

# Estado de autorización y estimación
autorizado = False
ultima_estimacion = {"sys": "---", "dia": "---", "nivel": "---"}

# Conexión a MySQL
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="",  # Cambia si tienes contraseña
    database="monitoreo_salud"
)
cursor = db.cursor()

@app.route("/")
def index():
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def registrar_datos():
    global autorizado, ultima_estimacion

    data = request.get_json()
    hr = int(data.get("hr", 0))
    spo2 = float(data.get("spo2", 0))
    ir = int(data.get("ir", 0))
    red = int(data.get("red", 0))
    id_paciente = int(data.get("id_paciente", 0))

    entrada = pd.DataFrame([[hr, spo2]], columns=["hr", "spo2"])
    sys = modelo_sys.predict(entrada)[0]
    dia = modelo_dia.predict(entrada)[0]

    # Clasificación del nivel de presión
    nivel = ""
    if sys >= 180 or dia >= 110:
        nivel = "H3 Alerta! ACV"
    elif sys >= 160 or dia >= 100:
        nivel = "H3"
    elif sys >= 140 or dia >= 90:
        nivel = "H2"
    elif sys >= 130 or dia >= 80:
        nivel = "H1"
    else:
        nivel = "Normal"

    ultima_estimacion = {
        "sys": f"{sys:.2f}",
        "dia": f"{dia:.2f}",
        "nivel": nivel
    }

    # Guardar en CSV solo si autorizado (no se toca este proceso)
    if autorizado:
        with open("registro_sensor_entrenamiento.csv", "a") as f:
            f.write(f"{hr},{spo2},{ir},{red},{sys:.2f},{dia:.2f}\n")

    # Guardar en MySQL si IR y RED son altos
    if ir > 20000 and red > 15000:
        sql = """
            INSERT INTO mediciones (id_paciente, fecha, hr, spo2, sys, dia, nivel)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        values = (id_paciente, datetime.now(), hr, spo2, sys, dia, nivel)
        cursor.execute(sql, values)
        db.commit()

    return jsonify({"sys": round(sys, 2), "dia": round(dia, 2), "nivel": nivel})

@app.route("/api/autorizacion", methods=["GET"])
def estado_autorizacion():
    return jsonify({"autorizado": autorizado})

@app.route("/api/autorizar", methods=["GET"])
def autorizar_registro():
    global autorizado
    autorizado = True
    return jsonify({"mensaje": "✅ Registro autorizado"})

@app.route("/api/detener", methods=["GET"])
def detener_registro():
    global autorizado
    autorizado = False
    return jsonify({"mensaje": "⛔ Registro detenido"})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
