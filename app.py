from flask import Flask, request, jsonify, render_template
import pandas as pd
import joblib
import numpy as np 
import os
import csv
from datetime import datetime
import mysql.connector
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaFileUpload
from scipy.signal import butter, filtfilt

# Importar la librería para hacer peticiones HTTP
import requests
from urllib.parse import quote # Para codificar el texto del mensaje

app = Flask(__name__)

# --- Carga de Modelos de Machine Learning ---
MODEL_SYS_PATH = "modelo_sys.pkl"
MODEL_DIA_PATH = "modelo_dia.pkl"
modelo_sys = None; modelo_dia = None
try:
    if os.path.exists(MODEL_SYS_PATH): modelo_sys = joblib.load(MODEL_SYS_PATH)
    if os.path.exists(MODEL_DIA_PATH): modelo_dia = joblib.load(MODEL_DIA_PATH)
    if modelo_sys and modelo_dia: print("✅ Modelos de ML cargados.")
    else: print("⚠️ Advertencia: Uno o ambos modelos de ML no se encontraron.")
except Exception as e: print(f"❌ Error al cargar modelos: {e}")

# --- Variables Globales ---
autorizado = False; capturando_entrenamiento = False; buffer_datos_entrenamiento = [] 
ultima_estimacion = {
    "sys": "---", "dia": "---", "spo2": "---", "hr": "---", "nivel": "---", 
    "timestamp": "---", "modo_autorizado": False, "capturando_entrenamiento": False,
    "raw_ir": "---", "raw_red": "---"
}
ventana_ir = []; ventana_red = []; MUESTRAS = 10 

# --- Configuración DB, Drive y CallMeBot ---
DB_HOST = os.environ.get('DB_HOST', os.environ.get("MYSQLHOST")) 
DB_USER = os.environ.get('DB_USER', os.environ.get("MYSQLUSER"))
DB_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get("MYSQLPASSWORD"))
DB_NAME = os.environ.get('DB_NAME', os.environ.get("MYSQLDATABASE"))
DB_PORT = os.environ.get('DB_PORT', os.environ.get("MYSQLPORT", "3306")) 
DB_CONFIG = {'host': DB_HOST, 'user': DB_USER, 'password': DB_PASSWORD, 'database': DB_NAME, 'port': int(DB_PORT) if DB_PORT and DB_PORT.isdigit() else 3306}

KEY_FILE_LOCATION_FALLBACK = 'service_account.json' 
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-") 
CSV_FILENAME = "registro_sensor_entrenamiento_alta_calidad.csv" 

CALLMEBOT_PHONE_NUMBER = os.environ.get('CALLMEBOT_PHONE_NUMBER')
CALLMEBOT_API_KEY = os.environ.get('CALLMEBOT_API_KEY')

# --- Funciones ---
def send_whatsapp_alert(message_body):
    if not all([CALLMEBOT_PHONE_NUMBER, CALLMEBOT_API_KEY]):
        print("⚠️ Advertencia CallMeBot: Faltan variables de entorno.")
        return False
    message_encoded = quote(message_body)
    url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE_NUMBER}&text={message_encoded}&apikey={CALLMEBOT_API_KEY}"
    try:
        print(f"Intentando enviar alerta vía CallMeBot...")
        response = requests.get(url, timeout=10)
        if response.status_code == 200: print(f"✅ Alerta de WhatsApp enviada a CallMeBot.")
        else: print(f"❌ Error al enviar WhatsApp con CallMeBot. Código: {response.status_code}")
        return response.status_code == 200
    except requests.exceptions.RequestException as e:
        print(f"❌ Excepción al enviar WhatsApp: {e}")
        return False

def clasificar_nivel_presion(pas_valor, pad_valor):
    if pas_valor is None or pd.isna(pas_valor): return "---"
    if pas_valor > 180 or pad_valor > 120: return "HT Crisis"
    elif pas_valor >= 140 or pad_valor >= 90: return "HT2"        
    elif (pas_valor >= 130 and pas_valor <= 139) or (pad_valor >= 80 and pad_valor <= 89): return "HT1"        
    elif (pas_valor >= 120 and pas_valor <= 129) and pad_valor < 80: return "Elevada"
    elif pas_valor < 120 and pad_valor < 80: return "Normal"
    else: return "Revisar" 

def conectar_db():
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]): print("Error: Configuración DB incompleta."); return None
    try: conn = mysql.connector.connect(**DB_CONFIG); return conn
    except mysql.connector.Error as err: print(f"❌ Error conexión MySQL: {err}"); return None

def guardar_medicion_mysql(id_paciente, sys, dia, nivel):
    conn = conectar_db(); 
    if conn is None: return False
    cursor = conn.cursor()
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    try:
        cursor.execute(query, (id_paciente, sys, dia, nivel)); conn.commit()
        print(f"MySQL: Datos guardados: {id_paciente}, {sys}, {dia}, {nivel}")
        return True
    except mysql.connector.Error as err: print(f"❌ Error MySQL: {err}"); conn.rollback(); return False
    finally:
        if conn.is_connected(): cursor.close(); conn.close()

# --- Endpoint Principal Modificado ---

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia, ventana_ir, ventana_red
    if not modelo_sys or not modelo_dia: return jsonify({"error": "Modelos ML no cargados"}), 500
    
    try:
        data = request.get_json();
        if not data: return jsonify({"error": "Request JSON"}), 400
        
        hr = int(data.get("hr", 0)); ir_val = int(data.get("ir", 0)); red_val = int(data.get("red", 0))
        id_paciente_in = int(data.get("id_paciente", 1)) 

        # --- LÓGICA DE CÁLCULO Y PREDICCIÓN RESTAURADA (COMO EN TU CÓDIGO ORIGEN) ---
        ventana_ir.append(ir_val)
        ventana_red.append(red_val)
        if len(ventana_ir) > MUESTRAS: ventana_ir.pop(0)
        if len(ventana_red) > MUESTRAS: ventana_red.pop(0)

        spo2_estimada = 0.0 
        if len(ventana_ir) == MUESTRAS:
            try:
                np_ventana_ir = np.array(ventana_ir); np_ventana_red = np.array(ventana_red)
                dc_ir = np.mean(np_ventana_ir); dc_red = np.mean(np_ventana_red)
                ac_ir = np.mean(np.abs(np_ventana_ir - dc_ir)) 
                ac_red = np.mean(np.abs(np_ventana_red - dc_red))
                if dc_ir > 0 and dc_red > 0: 
                    ratio = (ac_red / dc_red) / (ac_ir / dc_ir) if ac_ir > 0 and ac_red > 0 else 0
                    spo2_calc = 110 - 25 * ratio
                    spo2_estimada = max(70.0, min(100.0, spo2_calc))
            except Exception as e_spo2: print(f"Error en cálculo SpO2: {e_spo2}"); spo2_estimada = 0.0
        
        # Predecir SIEMPRE, incluso si spo2 es 0
        entrada_df = pd.DataFrame([[float(hr), float(spo2_estimada)]], columns=['hr', 'spo2'])
        sys_estimada = round(modelo_sys.predict(entrada_df)[0], 2)
        dia_estimada = round(modelo_dia.predict(entrada_df)[0], 2)
        nivel_presion = clasificar_nivel_presion(sys_estimada, dia_estimada)
        # --- FIN DE LA LÓGICA RESTAURADA ---
        
        ultima_estimacion = {
            "sys": f"{sys_estimada:.2f}", "dia": f"{dia_estimada:.2f}",
            "spo2": f"{spo2_estimada:.1f}", "hr": str(hr), "nivel": nivel_presion, 
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            "modo_autorizado": autorizado
            # No enviamos raw_ir/red para mantener la UI limpia como pediste
        }
        
        if nivel_presion == "HT Crisis":
            print("🚨 Detectada Crisis Hipertensiva, intentando enviar alerta por WhatsApp...")
            mensaje_alerta = (f"¡ALERTA MÉDICA URGENTE! 🚑\n"
                              f"Dispositivo del paciente ID {id_paciente_in} ha registrado una posible **Crisis Hipertensiva**.\n"
                              f"Valores Estimados: PAS={sys_estimada} mmHg, PAD={dia_estimada} mmHg.\n"
                              f"Timestamp: {ultima_estimacion['timestamp']} (UTC).\n"
                              f"**Verificar inmediatamente.**")
            send_whatsapp_alert(mensaje_alerta)
        
        # Guardado en Railway (tu lógica original de umbrales)
        if ir_val > 20000 and red_val > 15000: 
            guardar_medicion_mysql(id_paciente_in, sys_estimada, dia_estimada, nivel_presion)
        
        return jsonify({
            "sys": sys_estimada, "dia": dia_estimada, 
            "spo2": spo2_estimada, "nivel": nivel_presion
        }), 200
    except Exception as e: 
        print(f"❌ Error en /api/presion: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Error interno", "detalle": str(e)}), 500

# El resto de tus endpoints se mantienen igual (/, /ultimas_mediciones, /autorizacion, etc.)
# ...
```
*He omitido el resto de las funciones y endpoints que no necesitaban cambios para enfocarme en la corrección clave. Por favor, **reemplaza ÚNICAMENTE la función `api_procesar_presion` en tu archivo `app.py` existente**.*

**Qué hace este código corregido:**

* **Restaura tu Lógica Original:** Ahora, el código **siempre** intentará hacer una predicción de `sys_estimada` y `dia_estimada`, incluso si el cálculo de `spo2_estimada` da como resultado `0`. Esto significa que `sys_estimada` y `dia_estimada` siempre serán valores numéricos.
* **Muestra los Valores:** Como `sys` y `dia` ahora siempre tendrán un valor numérico, se mostrarán correctamente en la respuesta JSON, lo que a su vez hará que aparezcan en tu `index.html` y en tu pantalla LCD.
* **Habilita el Guardado en DB:** Como `pas_estimada` (`sys_estimada`) ya no será `-1`, la condición para guardar en la base de datos de Railway (`if ir_val > 20000 ...`) ahora solo dependerá de los umbrales de IR y RED, como en tu código original.

**Por favor, haz este cambio en tu `app.py`, despliégalo en Render y prueba de nuevo.** Ahora deberías ver los valores de PAS, PAD, SpO2 y Nivel actualizándose en tu página web y en la pantalla L
