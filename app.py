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
modelo_sys = None
modelo_dia = None
try:
    if os.path.exists(MODEL_SYS_PATH): modelo_sys = joblib.load(MODEL_SYS_PATH)
    if os.path.exists(MODEL_DIA_PATH): modelo_dia = joblib.load(MODEL_DIA_PATH)
    if modelo_sys and modelo_dia: print("✅ Modelos de ML cargados exitosamente.")
    else: print("⚠️ Advertencia: Uno o ambos modelos de ML no se encontraron.")
except Exception as e: print(f"❌ Error al cargar modelos: {e}")

# --- Variables Globales ---
autorizado = False 
capturando_entrenamiento = False 
buffer_datos_entrenamiento = [] 
ultima_estimacion = {
    "sys": "---", "dia": "---", "spo2": "---", "hr": "---", "nivel": "---", 
    "timestamp": "---", "modo_autorizado": False, "capturando_entrenamiento": False
}
ventana_ir = []
ventana_red = []
MUESTRAS = 10 

# --- Configuración DB y Drive ---
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

# --- Configuración de CallMeBot (desde variables de entorno) ---
CALLMEBOT_PHONE_NUMBER = os.environ.get('CALLMEBOT_PHONE_NUMBER') # Tu número, ej: 5939*******
CALLMEBOT_API_KEY = os.environ.get('CALLMEBOT_API_KEY') # La API Key que te dio el bot

# --- Funciones ---

def send_whatsapp_alert(message_body):
    """Envía una alerta por WhatsApp usando CallMeBot."""
    if not all([CALLMEBOT_PHONE_NUMBER, CALLMEBOT_API_KEY]):
        print("⚠️ Advertencia CallMeBot: Faltan variables de entorno CALLMEBOT_PHONE_NUMBER o CALLMEBOT_API_KEY. No se puede enviar WhatsApp.")
        return False
    
    # Codificar el mensaje para que sea seguro en una URL
    message_encoded = quote(message_body)
    
    # Construir la URL de la API de CallMeBot
    url = f"https://api.callmebot.com/whatsapp.php?phone={CALLMEBOT_PHONE_NUMBER}&text={message_encoded}&apikey={CALLMEBOT_API_KEY}"
    
    try:
        print(f"Intentando enviar alerta de WhatsApp a {CALLMEBOT_PHONE_NUMBER} vía CallMeBot...")
        response = requests.get(url, timeout=10) # Enviar la petición GET, con un timeout de 10 segundos
        
        # CallMeBot responde con texto plano, no JSON. El código 200 indica que la petición fue recibida.
        if response.status_code == 200:
            print(f"✅ Alerta de WhatsApp enviada a CallMeBot. Respuesta del servidor: {response.text}")
            return True
        else:
            print(f"❌ Error al enviar WhatsApp con CallMeBot. Código de estado: {response.status_code}, Respuesta: {response.text}")
            return False

    except requests.exceptions.RequestException as e:
        print(f"❌ Excepción al enviar WhatsApp con CallMeBot: {e}")
        return False

def filtrar_senal_ppg(senal, lowcut=0.5, highcut=5.0, fs=50.0, order=4):
    try:
        nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band'); return filtfilt(b, a, senal)
    except Exception: return np.array(senal)

def extraer_caracteristicas_ppg(segmento_ir, segmento_red, hr_promedio, spo2_promedio):
    ir_filtrado = filtrar_senal_ppg(segmento_ir)
    red_filtrado = filtrar_senal_ppg(segmento_red)
    caracteristicas = {
        "hr_promedio_sensor": round(hr_promedio, 2) if hr_promedio is not None else -1,
        "spo2_promedio_sensor": round(spo2_promedio, 2) if spo2_promedio is not None else -1,
        "ir_mean_filtrado": round(np.mean(ir_filtrado), 2) if len(ir_filtrado) > 0 else -1,
        "red_mean_filtrado": round(np.mean(red_filtrado), 2) if len(red_filtrado) > 0 else -1,
        "ir_std_filtrado": round(np.std(ir_filtrado), 2) if len(ir_filtrado) > 0 else -1,
        "red_std_filtrado": round(np.std(red_filtrado), 2) if len(red_filtrado) > 0 else -1
    }
    return caracteristicas

def clasificar_nivel_presion(pas_valor, pad_valor):
    if pas_valor is None or pd.isna(pas_valor) or pas_valor == -1: return "---"
    if pas_valor > 180 or pad_valor > 120: return "HT Crisis"
    elif pas_valor >= 140 or pad_valor >= 90: return "HT2"        
    elif (pas_valor >= 130 and pas_valor <= 139) or (pad_valor >= 80 and pad_valor <= 89): return "HT1"        
    elif (pas_valor >= 120 and pas_valor <= 129) and pad_valor < 80: return "Elevada"
    elif pas_valor < 120 and pad_valor < 80: return "Normal"
    else: 
        if pad_valor >= 80: 
             if pad_valor >= 90: return "HT2"
             else: return "HT1" 
        return "Revisar" 

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
        return True
    except mysql.connector.Error as err: print(f"❌ Error MySQL: {err}"); conn.rollback(); return False
    finally:
        if conn.is_connected(): cursor.close(); conn.close()

def get_google_drive_service():
    effective_key_file_location = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', KEY_FILE_LOCATION_FALLBACK)
    if not os.path.exists(effective_key_file_location): print(f"⚠️ Drive: '{effective_key_file_location}' no encontrado."); return None
    try:
        creds = service_account.Credentials.from_service_account_file(effective_key_file_location, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds); return service
    except Exception as e: print(f"❌ Drive: Error autenticando: {e}"); return None

def subir_archivo_a_drive(file_path, file_name_on_drive):
    if not FOLDER_ID or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
        print("Error Drive: FOLDER_ID no configurado."); return False
    service = get_google_drive_service();
    if not service: return False
    try:
        file_metadata = {'name': file_name_on_drive, 'parents': [FOLDER_ID]}
        media = MediaFileUpload(file_path, mimetype='text/csv', resumable=True)
        query = f"name='{file_name_on_drive}' and '{FOLDER_ID}' in parents and trashed=false"
        response = service.files().list(q=query, spaces='drive', fields='files(id, name)').execute()
        existing_files = response.get('files', [])
        if existing_files: 
            file_id = existing_files[0].get('id')
            service.files().update(fileId=file_id, media_body=media).execute()
        else: 
            service.files().create(body=file_metadata, media_body=media, fields='id').execute()
        print(f"Drive: Archivo '{file_name_on_drive}' actualizado/subido.")
        return True
    except Exception as e: print(f"❌ Drive: Error al subir: {e}"); return False

# --- Endpoints Flask ---

@app.route("/")
def home():
    global ultima_estimacion, autorizado, capturando_entrenamiento
    ultima_estimacion["modo_autorizado"] = autorizado 
    ultima_estimacion["capturando_entrenamiento"] = capturando_entrenamiento
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia, ventana_ir, ventana_red, capturando_entrenamiento, buffer_datos_entrenamiento
    if not modelo_sys or not modelo_dia: return jsonify({"error": "Modelos ML no cargados"}), 500
    
    try:
        data = request.get_json();
        if not data: return jsonify({"error": "Request JSON"}), 400
        
        hr_in = data.get("hr", 75); ir_in = data.get("ir", 0); red_in = data.get("red", 0)
        id_paciente_in = data.get("id_paciente", 99) 
        test_pas = data.get("test_pas"); test_pad = data.get("test_pad")

        if test_pas is not None and test_pad is not None:
            print(f"🧪 MODO PRUEBA: Usando valores PAS/PAD manuales: {test_pas}/{test_pad}")
            sys_estimada = float(test_pas); dia_estimada = float(test_pad)
            spo2_estimada = 98.0; hr = float(hr_in)
        else:
            try: hr = int(hr_in); ir_val = int(ir_in); red_val = int(red_in) 
            except ValueError: return jsonify({"error": "Datos deben ser numéricos"}), 400
            
            # Restaurada la lógica de tu código origen para SpO2 y predicción
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
            
            entrada_df = pd.DataFrame([[float(hr), float(spo2_estimada)]], columns=['hr', 'spo2'])
            sys_estimada = round(modelo_sys.predict(entrada_df)[0], 2)
            dia_estimada = round(modelo_dia.predict(entrada_df)[0], 2)

        nivel_presion = clasificar_nivel_presion(sys_estimada, dia_estimada)
        
        ultima_estimacion = {
            "sys": f"{sys_estimada:.2f}", "dia": f"{dia_estimada:.2f}",
            "spo2": f"{spo2_estimada:.1f}", "hr": str(hr), "nivel": nivel_presion, 
            "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S'), 
            "modo_autorizado": autorizado, "capturando_entrenamiento": capturando_entrenamiento
        }

        if nivel_presion == "HT Crisis":
            print("🚨 Detectada Crisis Hipertensiva, intentando enviar alerta por WhatsApp...")
            mensaje_alerta = (f"¡ALERTA MÉDICA URGENTE! 🚑\n\n"
                              f"Dispositivo del paciente ID {id_paciente_in} ha registrado una posible **Crisis Hipertensiva**.\n\n"
                              f"Valores Registrados:\n"
                              f"  - *PAS:* {sys_estimada} mmHg\n"
                              f"  - *PAD:* {dia_estimada} mmHg\n\n"
                              f"Timestamp: {ultima_estimacion['timestamp']} (UTC).\n\n"
                              f"**Verificar inmediatamente.**")
            send_whatsapp_alert(mensaje_alerta)
        
        if (data.get("ir", 0) > 20000 and data.get("red", 0) > 15000) or (test_pas is not None): 
            guardar_medicion_mysql(id_paciente_in, sys_estimada, dia_estimada, nivel_presion)
        
        return jsonify({
            "sys": sys_estimada, "dia": dia_estimada, 
            "spo2": spo2_estimada, "nivel": nivel_presion
        }), 200
    except Exception as e: 
        print(f"❌ Error en /api/presion: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Error interno", "detalle": str(e)}), 500

# ... (El resto de tus endpoints: /iniciar_captura..., /detener_captura..., /guardar_muestra..., 
#      /ultimas_mediciones, /autorizacion, /ultima_estimacion se mantienen igual que en la versión anterior) ...

# ... (El bloque if __name__ == "__main__": se mantiene igual) ...
```
*Nota: Para mantener la respuesta enfocada, he omitido las funciones y endpoints que no necesitaban cambios. Debes **integrar la nueva función `send_whatsapp_alert` y la nueva lógica de configuración de CallMeBot en tu archivo `app.py` completo**, así como la lógica para manejar los valores de prueba en el endpoint `/api/presion
