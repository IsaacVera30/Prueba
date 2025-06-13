import eventlet
eventlet.monkey_patch()

from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import pandas as pd
import joblib
import os
import mysql.connector
import requests
from datetime import datetime
import traceback
import time # Añadido para el temporizador

app = Flask(__name__)
socketio = SocketIO(app)

# --- Carga de Modelos y Configuración ---
try:
    modelo_sys = joblib.load('modelo_sys.pkl')
    modelo_dia = joblib.load('modelo_dia.pkl')
    print("✅ Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR: No se pudieron cargar los modelos de ML: {e}")
    modelo_sys, modelo_dia = None, None

DB_CONFIG = {
    'host': os.environ.get("MYSQLHOST"), 'user': os.environ.get("MYSQLUSER"),
    'password': os.environ.get("MYSQLPASSWORD"), 'database': os.environ.get("MYSQLDATABASE"),
    'port': int(os.environ.get("MYSQLPORT", "3306"))
}
autorizado = False
last_db_save_time = 0 # Variable para controlar el tiempo del último guardado

# --- Funciones Auxiliares ---
# (Las funciones conectar_db, guardar_medicion_mysql, enviar_alerta_whatsapp, y clasificar_nivel_presion se mantienen igual)
# ...

# --- Rutas de la API ---
@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global last_db_save_time # Se usa la variable global
    
    if not modelo_sys or not modelo_dia:
        return jsonify({"error": "Modelos no disponibles"}), 503
    try:
        data = request.get_json()
        
        hr_promedio = float(data["hr_promedio"])
        spo2_sensor = float(data["spo2_sensor"])
        id_paciente = int(data.get("id_paciente", 1))

        entrada_df = pd.DataFrame([[hr_promedio, spo2_sensor]], columns=['hr', 'spo2'])
        pas_estimada = modelo_sys.predict(entrada_df)[0]
        pad_estimada = modelo_dia.predict(entrada_df)[0]
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)

        datos_para_panel = {
            "hr_crudo": data.get("hr_crudo"), "hr_promedio": f"{hr_promedio:.0f}",
            "spo2_sensor": f"{spo2_sensor:.1f}", "ir": data.get("ir"), "red": data.get("red"),
            "sys_ml": f"{pas_estimada:.2f}", "dia_ml": f"{pad_estimada:.2f}",
            "hr_ml": f"{hr_promedio:.0f}", "spo2_ml": f"{spo2_sensor:.1f}", "estado": nivel_presion
        }
        socketio.emit('update_data', datos_para_panel)

        # --- LÓGICA DE GUARDADO Y ALERTA ---
        
        # 1. Lógica de Alerta (se ejecuta siempre)
        if nivel_presion == "HT Crisis":
            enviar_alerta_whatsapp(nivel_presion, pas_estimada, pad_estimada)

        # 2. Lógica de Guardado (solo si está autorizado y han pasado 5 segundos)
        if autorizado:
            current_time = time.time()
            if (current_time - last_db_save_time) >= 5:
                guardar_medicion_mysql(id_paciente, pas_estimada, pad_estimada, nivel_presion, hr_promedio, spo2_sensor)
                socketio.emit('new_record_saved') # Notifica al panel que la tabla debe actualizarse
                last_db_save_time = current_time # Actualiza el tiempo del último guardado
                print("✅ Datos guardados en BD (temporizador de 5s cumplido).")

        return jsonify({ "sys": pas_estimada, "dia": pad_estimada, "hr": hr_promedio, "spo2": spo2_sensor, "nivel": nivel_presion }), 200
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

# (El resto de las rutas y el if __name__ == "__main__" se mantienen exactamente igual)
# ...
