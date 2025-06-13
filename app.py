# AÑADIDO: Líneas para activar eventlet ANTES de todo lo demás. Es crucial que estén primero.
import eventlet
eventlet.monkey_patch()

# El resto de tus imports van DESPUÉS de esas dos líneas
from flask import Flask, request, jsonify, render_template
from flask_socketio import SocketIO
import pandas as pd
import joblib
import os
import mysql.connector
from datetime import datetime
import traceback

app = Flask(__name__)
socketio = SocketIO(app)

# Carga de modelos del repositorio actual
try:
    modelo_sys = joblib.load('modelo_sys.pkl')
    modelo_dia = joblib.load('modelo_dia.pkl')
    print("✅ Modelos de ML cargados correctamente.")
except Exception as e:
    print(f"❌ ERROR: No se pudieron cargar los modelos de ML: {e}")
    modelo_sys, modelo_dia = None, None

# Configuración de Base de Datos
DB_CONFIG = {
    'host': os.environ.get("MYSQLHOST"),
    'user': os.environ.get("MYSQLUSER"),
    'password': os.environ.get("MYSQLPASSWORD"),
    'database': os.environ.get("MYSQLDATABASE"),
    'port': int(os.environ.get("MYSQLPORT", "3306"))
}
autorizado = False  # Estado para controlar el guardado en BD

def conectar_db():
    try:
        return mysql.connector.connect(**DB_CONFIG)
    except Exception as e:
        print(f"❌ Error DB: {e}")
        return None

def guardar_medicion_mysql(id_paciente, sys, dia, nivel):
    conn = conectar_db()
    if not conn:
        return
    cursor = conn.cursor()
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    try:
        cursor.execute(query, (id_paciente, sys, dia, nivel))
        conn.commit()
    except Exception as e:
        print(f"❌ Error al guardar en MySQL: {e}")
    finally:
        if conn.is_connected():
            conn.close()

def clasificar_nivel_presion(pas, pad):
    if pas is None or pad is None:
        return "N/A"
    if pas > 180 or pad > 120: return "HT Crisis"
    if pas >= 140 or pad >= 90: return "HT2"
    if (pas >= 130 and pas <= 139) or (pad >= 80 and pad <= 89): return "HT1"
    if (pas >= 120 and pas <= 129) and pad < 80: return "Elevada"
    return "Normal"

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    if not modelo_sys or not modelo_dia:
        return jsonify({"error": "Modelos no disponibles en el servidor"}), 503

    try:
        data = request.get_json()
        hr_promedio = float(data["hr_promedio"])
        ir_val = float(data["ir"])
        red_val = float(data["red"])
        id_paciente = int(data.get("id_paciente", 1))

        entrada_df = pd.DataFrame([[hr_promedio, ir_val, red_val]], columns=['HR', 'IR', 'RED'])
        pas_estimada = modelo_sys.predict(entrada_df)[0]
        pad_estimada = modelo_dia.predict(entrada_df)[0]
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)

        datos_para_panel = {
            "hr_crudo": data.get("hr_crudo"), "hr_promedio": f"{hr_promedio:.0f}",
            "spo2_sensor": data.get("spo2_sensor"), "ir": f"{ir_val:.0f}", "red": f"{red_val:.0f}",
            "sys_ml": f"{pas_estimada:.2f}", "dia_ml": f"{pad_estimada:.2f}",
            "hr_ml": f"{hr_promedio:.0f}", "spo2_ml": data.get("spo2_sensor"), "estado": nivel_presion
        }
        
        socketio.emit('update_data', datos_para_panel)

        if autorizado:
            guardar_medicion_mysql(id_paciente, pas_estimada, pad_estimada, nivel_presion)
            socketio.emit('new_record_saved')

        respuesta_para_dispositivo = {
            "sys": pas_estimada, "dia": pad_estimada, "hr": hr_promedio,
            "spo2": float(data.get("spo2_sensor", 0)), "nivel": nivel_presion
        }
        return jsonify(respuesta_para_dispositivo), 200

    except Exception as e:
        print(f"❌❌❌ ERROR CATASTRÓFICO en /api/presion: {e} ❌❌❌")
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


@app.route("/api/autorizacion", methods=["GET", "POST"])
def api_control_autorizacion():
    global autorizado
    if request.method == "POST":
        autorizado = request.json.get("autorizado", False)
        print(f"Estado de autorización cambiado a: {autorizado}")
        socketio.emit('status_update', {"autorizado": autorizado})
    return jsonify({"autorizado": autorizado})

@app.route("/api/ultimas_mediciones")
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if not conn:
        return jsonify([])
    cursor = conn.cursor(dictionary=True)
    query = "SELECT id, id_paciente, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
    try:
        cursor.execute(query)
        records = cursor.fetchall()
        for rec in records:
            for key in rec:
                rec[key] = str(rec[key])
        conn.close()
        return jsonify(records)
    except Exception as e:
        print(f"❌ Error al leer historial de DB: {e}")
        if conn.is_connected():
            conn.close()
        return jsonify([])

if __name__ == "__main__":
    print("Iniciando servidor Flask con SocketIO y Eventlet...")
    port = int(os.environ.get("PORT", 10000))
    # Esta línea es usada para desarrollo local, gunicorn la ignora en producción pero es buena práctica tenerla
    socketio.run(app, host='0.0.0.0', port=port)
