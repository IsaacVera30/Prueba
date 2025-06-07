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
from scipy.signal import butter, filtfilt # Para filtrar señales PPG

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
autorizado = False # Para el modo general de registro CSV a Drive
capturando_entrenamiento = False # Nuevo flag para la captura específica de muestras de entrenamiento
buffer_datos_entrenamiento = [] # Buffer para almacenar datos del sensor durante la captura
MAX_BUFFER_ENTRENAMIENTO = 250 # Ej: 50 Hz * 5 segundos (ajusta según tu frecuencia de muestreo y tiempo deseado)

ultima_estimacion = {
    "sys": "---", "dia": "---", "spo2": "---", 
    "hr": "---", "nivel": "---", "timestamp": "---", "modo_autorizado": False
}
ventana_ir = [] # Para cálculo de SpO2 en tiempo real
ventana_red = [] # Para cálculo de SpO2 en tiempo real
MUESTRAS_SPO2 = 10 

# --- Configuración DB y Drive (sin cambios respecto a la última versión) ---
DB_HOST = os.environ.get('DB_HOST', os.environ.get("MYSQLHOST")) 
DB_USER = os.environ.get('DB_USER', os.environ.get("MYSQLUSER"))
DB_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get("MYSQLPASSWORD"))
DB_NAME = os.environ.get('DB_NAME', os.environ.get("MYSQLDATABASE"))
DB_PORT = os.environ.get('DB_PORT', os.environ.get("MYSQLPORT", "3306")) 
DB_CONFIG = {'host': DB_HOST, 'user': DB_USER, 'password': DB_PASSWORD, 'database': DB_NAME, 'port': int(DB_PORT) if DB_PORT and DB_PORT.isdigit() else 3306}
print(f"Configuración DB: {DB_CONFIG['host']}, {DB_CONFIG['user']}, {DB_CONFIG['database']}, {DB_CONFIG['port']}")

KEY_FILE_LOCATION_FALLBACK = 'service_account.json' 
SCOPES = ['https://www.googleapis.com/auth/drive.file']
FOLDER_ID = os.environ.get('GOOGLE_DRIVE_FOLDER_ID', "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-") 
CSV_FILENAME = "registro_sensor_entrenamiento.csv" # Este será el CSV de ALTA CALIDAD
if not FOLDER_ID or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
    print("⚠️ Advertencia: GOOGLE_DRIVE_FOLDER_ID no está bien configurado.")

# --- Funciones de Procesamiento de Señal y Características PPG (Ejemplos) ---
def filtrar_senal_ppg(senal, lowcut=0.5, highcut=5.0, fs=50.0, order=4):
    """Aplica un filtro Butterworth pasabanda a la señal PPG."""
    # fs: frecuencia de muestreo de tu sensor (ej. 50 Hz)
    # lowcut, highcut: frecuencias de corte para el filtro
    try:
        nyquist = 0.5 * fs
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        senal_filtrada = filtfilt(b, a, senal)
        return senal_filtrada
    except Exception as e:
        print(f"Error al filtrar señal: {e}")
        return np.array(senal) # Devolver original si falla el filtro

def extraer_caracteristicas_ppg(segmento_ir, segmento_red, hr_promedio, spo2_promedio):
    """
    Extrae características representativas de un segmento de señales PPG.
    ¡ESTA FUNCIÓN ES CLAVE Y NECESITAS DESARROLLARLA CON CARACTERÍSTICAS ÚTILES!
    """
    # Placeholder: Aquí deberías implementar la extracción de características robustas.
    # Ejemplos muy básicos (necesitarás investigar e implementar más):
    # - Amplitud promedio de la señal IR y RED (después de filtrar)
    # - Variabilidad de la HR en el segmento (SDNN si tienes los tiempos entre latidos)
    # - Características de la forma de la onda (tiempos de subida, bajada, anchos de pico, etc.)
    # Por ahora, solo devolvemos HR y SpO2 promedio, y las señales crudas promediadas.
    
    # Filtrar señales
    ir_filtrado = filtrar_senal_ppg(segmento_ir)
    red_filtrado = filtrar_senal_ppg(segmento_red)

    caracteristicas = {
        "hr_promedio": round(hr_promedio, 2) if hr_promedio is not None else -1,
        "spo2_promedio": round(spo2_promedio, 2) if spo2_promedio is not None else -1,
        "ir_mean_filtrado": round(np.mean(ir_filtrado), 2) if len(ir_filtrado) > 0 else -1,
        "red_mean_filtrado": round(np.mean(red_filtrado), 2) if len(red_filtrado) > 0 else -1,
        "ir_std_filtrado": round(np.std(ir_filtrado), 2) if len(ir_filtrado) > 0 else -1,
        "red_std_filtrado": round(np.std(red_filtrado), 2) if len(red_filtrado) > 0 else -1
        # ... Añade aquí tus características PPG más avanzadas ...
    }
    return caracteristicas

# --- Funciones existentes (clasificar_nivel_presion, conectar_db, etc.) ---
def calcular_spo2_desde_origen(current_ir_list, current_red_list): # Modificado para tomar listas
    # Usa una copia de las listas para no modificar las globales si no es necesario
    # O ajusta para que tome las ventanas globales directamente.
    # Por ahora, asumimos que recibe las ventanas actuales.
    if not current_ir_list or not current_red_list or len(current_ir_list) < MUESTRAS_SPO2:
        return 0.0 

    # Usar solo las últimas MUESTRAS_SPO2 para el cálculo si las listas son más largas
    ir_window_calc = np.array(current_ir_list[-MUESTRAS_SPO2:])
    red_window_calc = np.array(current_red_list[-MUESTRAS_SPO2:])
    
    spo2 = 0.0 
    try:
        dc_ir = np.mean(ir_window_calc); dc_red = np.mean(red_window_calc)
        ac_ir = np.mean(np.abs(ir_window_calc - dc_ir)) 
        ac_red = np.mean(np.abs(red_window_calc - dc_red))
        if dc_ir > 0 and dc_red > 0 and ac_ir > 0 and ac_red > 0 : 
            ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
            spo2_calc = 110 - 25 * ratio
            spo2 = max(70.0, min(100.0, spo2_calc))
        else: print(f"DEBUG: SpO2 (origen) - DC/AC inválido.")
    except Exception as e_spo2: print(f"Error en cálculo SpO2 (origen): {e_spo2}"); spo2 = 0.0             
    return round(spo2, 1)

def clasificar_nivel_presion(pas_valor, pad_valor): # Sin cambios
    if pas_valor is None or pad_valor is None or pd.isna(pas_valor) or pd.isna(pad_valor) or pas_valor == -1 or pad_valor == -1: return "---"
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

def conectar_db(): # Sin cambios
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]): print("Error: Configuración DB incompleta."); return None
    try: conn = mysql.connector.connect(**DB_CONFIG); print("✅ Conexión MySQL OK."); return conn
    except mysql.connector.Error as err: print(f"❌ Error conexión MySQL: {err}"); return None

def guardar_medicion_mysql(id_paciente_recibido, valor_pas_estimada, valor_pad_estimada, valor_nivel_calculado): # Sin cambios (guarda solo 4 campos)
    conn = conectar_db(); 
    if conn is None: print("guardar_med_mysql: No DB con."); return False
    cursor = conn.cursor()
    query = "INSERT INTO mediciones (id_paciente, sys, dia, nivel) VALUES (%s, %s, %s, %s)"
    datos_a_insertar = (id_paciente_recibido, valor_pas_estimada, valor_pad_estimada, valor_nivel_calculado)
    try:
        print(f"MySQL (simpl.): Insertando: {datos_a_insertar}")
        cursor.execute(query, datos_a_insertar); conn.commit()
        print(f"MySQL (simpl.): Datos OK: {datos_a_insertar}")
        return True
    except mysql.connector.Error as err: print(f"❌ Error MySQL (simpl.): {err}"); conn.rollback(); return False
    finally:
        if conn.is_connected(): cursor.close(); conn.close(); print("MySQL (simpl.): Conexión cerrada.")

def get_google_drive_service(): # Sin cambios
    effective_key_file_location = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', KEY_FILE_LOCATION_FALLBACK)
    print(f"Drive: Intentando creds desde: {effective_key_file_location}")
    if not os.path.exists(effective_key_file_location): print(f"⚠️ Drive: '{effective_key_file_location}' no encontrado."); return None
    try:
        creds = service_account.Credentials.from_service_account_file(effective_key_file_location, scopes=SCOPES)
        service = build('drive', 'v3', credentials=creds); print("✅ Drive: Servicio autenticado.")
        return service
    except Exception as e: print(f"❌ Drive: Error autenticando: {e}"); return None

def subir_archivo_a_drive(file_path, file_name_on_drive): # Sin cambios
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive_placeholder' or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
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
            updated_file = service.files().update(fileId=file_id, media_body=media).execute()
            print(f"Drive: Archivo '{file_name_on_drive}' actualizado. ID: {updated_file.get('id')}")
        else: 
            file = service.files().create(body=file_metadata, media_body=media, fields='id').execute()
            print(f"Drive: Archivo '{file_name_on_drive}' subido. ID: {file.get('id')}")
        return True
    except Exception as e: print(f"❌ Drive: Error al subir/actualizar: {e}"); return False

# --- Endpoints Flask ---
@app.route("/")
def home(): # Sin cambios
    global ultima_estimacion, autorizado
    ultima_estimacion["modo_autorizado"] = autorizado 
    return render_template("index.html", autorizado=autorizado, estimacion=ultima_estimacion)

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia, ventana_ir, ventana_red, capturando_entrenamiento, buffer_datos_entrenamiento
    
    if not modelo_sys or not modelo_dia: return jsonify({"error": "Modelos ML no cargados"}), 500
    try:
        data = request.get_json();
        if not data: return jsonify({"error": "Request JSON"}), 400
        hr_in = data.get("hr"); ir_in = data.get("ir"); red_in = data.get("red")
        id_paciente_in = data.get("id_paciente", 1) 
        if hr_in is None or ir_in is None or red_in is None: return jsonify({"error": "Datos incompletos"}), 400
        try: hr = int(hr_in); ir_val = int(ir_in); red_val = int(red_in) 
        except ValueError: return jsonify({"error": "Datos deben ser numéricos"}), 400

        # Lógica para el buffer de entrenamiento si está activada la captura
        if capturando_entrenamiento:
            if len(buffer_datos_entrenamiento) < MAX_BUFFER_ENTRENAMIENTO:
                buffer_datos_entrenamiento.append({"hr": hr, "ir": ir_val, "red": red_val, "timestamp": datetime.now()})
            else:
                print("Buffer de entrenamiento lleno. Detenga la captura para procesar.")
        
        # Estimación en tiempo real (siempre se hace, para `ultima_estimacion`)
        # Actualizar ventanas para SpO2 en tiempo real
        ventana_ir.append(ir_val) 
        ventana_red.append(red_val)
        if len(ventana_ir) > MUESTRAS_SPO2: ventana_ir.pop(0)
        if len(ventana_red) > MUESTRAS_SPO2: ventana_red.pop(0)
        
        spo2_estimada_rt = calcular_spo2_desde_origen(ventana_ir, ventana_red) # Usa las ventanas globales para SpO2 RT

        pas_estimada = -1.0; pad_estimada = -1.0 
        if spo2_estimada_rt > 0 and modelo_sys and modelo_dia: 
            entrada_df = pd.DataFrame([[float(hr), float(spo2_estimada_rt)]], columns=['hr', 'spo2'])
            pas_estimada = round(modelo_sys.predict(entrada_df)[0], 2)
            pad_estimada = round(modelo_dia.predict(entrada_df)[0], 2)
            
        nivel_presion = clasificar_nivel_presion(pas_estimada, pad_estimada)
        
        current_timestamp_str = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        ultima_estimacion = {
            "sys": f"{pas_estimada:.2f}" if pas_estimada != -1 else "---",
            "dia": f"{pad_estimada:.2f}" if pad_estimada != -1 else "---",
            "spo2": f"{spo2_estimada_rt:.1f}" if spo2_estimada_rt > 0 else "---",
            "hr": str(hr), "nivel": nivel_presion, "timestamp": current_timestamp_str, 
            "modo_autorizado": autorizado, "capturando_entrenamiento": capturando_entrenamiento
        }

        # Guardado en MySQL (solo estimaciones, no datos de entrenamiento)
        print(f"DEBUG: Antes de guardar en DB - ir_val={ir_val}, red_val={red_val}, pas_estimada={pas_estimada}")
        if ir_val > 20000 and red_val > 15000 and pas_estimada != -1: 
            guardar_medicion_mysql(id_paciente_in, pas_estimada, pad_estimada, nivel_presion)
        else: print("DEBUG: Condición para guardar en DB no cumplida.")

        return jsonify({
            "sys": pas_estimada, "dia": pad_estimada, 
            "spo2": spo2_estimada_rt, "nivel": nivel_presion
        }), 200
    except Exception as e:
        print(f"❌ Error general en /api/presion: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Error interno servidor", "detalle": str(e)}), 500

# --- NUEVOS ENDPOINTS PARA CAPTURA DE ENTRENAMIENTO ---
@app.route("/api/iniciar_captura_entrenamiento", methods=["POST"])
def iniciar_captura():
    global capturando_entrenamiento, buffer_datos_entrenamiento, autorizado
    if not autorizado: # Solo permitir iniciar captura si el modo general de registro está autorizado
        return jsonify({"error": "El registro general debe estar autorizado para iniciar captura de entrenamiento."}), 403
    
    capturando_entrenamiento = True
    buffer_datos_entrenamiento = [] # Limpiar buffer anterior
    print("✅ Captura de datos de entrenamiento INICIADA.")
    return jsonify({"mensaje": "Captura de entrenamiento iniciada.", "capturando": capturando_entrenamiento}), 200

@app.route("/api/guardar_muestra_entrenamiento", methods=["POST"])
def guardar_muestra():
    global capturando_entrenamiento, buffer_datos_entrenamiento, autorizado

    if not autorizado:
        return jsonify({"error": "El registro general no está autorizado."}), 403
    if not capturando_entrenamiento:
        return jsonify({"error": "La captura de entrenamiento no está activa."}), 400
    if not buffer_datos_entrenamiento:
        return jsonify({"error": "No hay datos en el buffer para guardar."}), 400

    try:
        data_referencia = request.get_json()
        pas_ref = float(data_referencia.get("pas_referencia"))
        pad_ref = float(data_referencia.get("pad_referencia"))
        
        if pas_ref is None or pad_ref is None:
            return jsonify({"error": "Valores de PAS y PAD de referencia son requeridos."}), 400

        # Procesar el buffer_datos_entrenamiento para extraer características
        # Por ejemplo, promediar HR, SpO2 y extraer características PPG del segmento
        if buffer_datos_entrenamiento:
            hrs_segmento = [d["hr"] for d in buffer_datos_entrenamiento]
            irs_segmento = [d["ir"] for d in buffer_datos_entrenamiento]
            reds_segmento = [d["red"] for d in buffer_datos_entrenamiento]
            
            hr_promedio_segmento = np.mean(hrs_segmento) if hrs_segmento else -1
            # Calcular SpO2 promedio para el segmento completo
            spo2_promedio_segmento = calcular_spo2_desde_origen(irs_segmento, reds_segmento) 
            
            # Extraer características PPG más avanzadas del segmento
            caracteristicas_extraidas = extraer_caracteristicas_ppg(irs_segmento, reds_segmento, hr_promedio_segmento, spo2_promedio_segmento)
            
            timestamp_captura = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Guardar en CSV
            with open(CSV_FILENAME, mode='a', newline='') as file_csv:
                writer = csv.writer(file_csv)
                # Encabezado (si el archivo es nuevo y está vacío)
                if os.stat(CSV_FILENAME).st_size == 0:
                    header = list(caracteristicas_extraidas.keys()) + ["pas_referencia", "pad_referencia", "timestamp_captura"]
                    writer.writerow(header)
                
                row_data = list(caracteristicas_extraidas.values()) + [pas_ref, pad_ref, timestamp_captura]
                writer.writerow(row_data)
            print(f"Muestra de entrenamiento guardada en {CSV_FILENAME}")

            if FOLDER_ID and FOLDER_ID != 'tu_id_de_carpeta_en_google_drive_placeholder' and FOLDER_ID != "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-":
                subir_archivo_a_drive(CSV_FILENAME, CSV_FILENAME)
        
        capturando_entrenamiento = False # Detener captura después de guardar
        buffer_datos_entrenamiento = [] # Limpiar buffer
        return jsonify({"mensaje": "Muestra de entrenamiento guardada exitosamente."}), 200

    except Exception as e:
        print(f"❌ Error al guardar muestra de entrenamiento: {e}")
        import traceback; traceback.print_exc()
        capturando_entrenamiento = False # Asegurar que se detenga la captura en caso de error
        buffer_datos_entrenamiento = []
        return jsonify({"error": "Error al procesar la muestra", "detalle": str(e)}), 500

# --- Fin Nuevos Endpoints ---
# NUEVO ENDPOINT PARA DETENER LA CAPTURA DESDE LA UI
@app.route("/api/detener_captura_entrenamiento", methods=["POST"])
def detener_captura():
    global capturando_entrenamiento
    capturando_entrenamiento = False
    print("✅ Captura de datos de entrenamiento DETENIDA por el cliente.")
    return jsonify({"mensaje": "Captura de entrenamiento detenida.", "capturando": capturando_entrenamiento}), 200
    

@app.route("/api/ultimas_mediciones", methods=["GET"]) # Sin cambios
def get_ultimas_mediciones_db():
    conn = conectar_db()
    if conn is None: return jsonify({"error": "No DB con"}), 500
    cursor = conn.cursor(dictionary=True) 
    query = "SELECT id, id_paciente, sys, dia, nivel FROM mediciones ORDER BY id DESC LIMIT 20"
    mediciones = []
    try:
        cursor.execute(query); mediciones = cursor.fetchall()
        print(f"get_ultimas_mediciones_db: {len(mediciones)} obtenidas.")
    except mysql.connector.Error as err: print(f"❌ Error MySQL (ultimas_med): {err}"); return jsonify({"error": "Error DB", "detalle": str(err)}), 500
    finally:
        if conn.is_connected(): cursor.close(); conn.close(); print("MySQL (ultimas_med): Conexión cerrada.")
    return jsonify(mediciones), 200

@app.route("/api/autorizacion", methods=["GET", "POST"]) # Sin cambios
def api_control_autorizacion():
    global autorizado, ultima_estimacion, capturando_entrenamiento
    if request.method == "POST":
        try:
            data = request.get_json()
            if data is None or "autorizado" not in data: return jsonify({"error": "Payload JSON esperado"}), 400
            nuevo_estado = data.get("autorizado")
            if isinstance(nuevo_estado, bool):
                autorizado = nuevo_estado
                print(f"Estado de autorización general cambiado a: {autorizado}")
                if not autorizado: # Si se detiene el registro general, también detener captura de entrenamiento
                    capturando_entrenamiento = False
                    buffer_datos_entrenamiento = []
                    print("Captura de entrenamiento detenida debido a detención de registro general.")
                ultima_estimacion["modo_autorizado"] = autorizado 
                ultima_estimacion["capturando_entrenamiento"] = capturando_entrenamiento
                return jsonify({"mensaje": f"Autorización general: {autorizado}, Captura entrenamiento: {capturando_entrenamiento}", "autorizado": autorizado, "capturando_entrenamiento": capturando_entrenamiento}), 200
            else: return jsonify({"error": "Valor 'autorizado' debe ser booleano"}), 400
        except Exception as e: print(f"❌ Error POST /api/autorizacion: {e}"); return jsonify({"error": "Error interno", "detalle": str(e)}), 400
    else: # GET
        return jsonify({"autorizado": autorizado, "capturando_entrenamiento": capturando_entrenamiento}), 200

@app.route("/api/ultima_estimacion", methods=["GET"]) # Sin cambios
def get_ultima_medicion():
    global ultima_estimacion, autorizado, capturando_entrenamiento
    ultima_estimacion["modo_autorizado"] = autorizado 
    ultima_estimacion["capturando_entrenamiento"] = capturando_entrenamiento
    return jsonify(ultima_estimacion)

if __name__ == "__main__":
    print("Iniciando servidor Flask para desarrollo local...")
    if not all([DB_CONFIG['host'], DB_CONFIG['user'], DB_CONFIG['database']]): print("ADVERTENCIA: Faltan variables DB.")
    if not FOLDER_ID or FOLDER_ID == 'tu_id_de_carpeta_en_google_drive_placeholder' or FOLDER_ID == "1tYCn9x-fDQUkHTOSNClGKtYU0Yov2OM-": print("ADVERTENCIA: GOOGLE_DRIVE_FOLDER_ID no configurado.")
    effective_key_file_location = os.environ.get('GOOGLE_APPLICATION_CREDENTIALS', KEY_FILE_LOCATION_FALLBACK)
    if not os.path.exists(effective_key_file_location): print(f"ADVERTENCIA: Credenciales '{effective_key_file_location}' no encontradas.")
    port = int(os.environ.get("PORT", 10000)) 
    app.run(host='0.0.0.0', port=port, debug=True)
# AÑADIR ESTA RUTA NUEVA TEMPORALMENTE EN app.py
# ¡¡¡ADVERTENCIA: ELIMINAR ESTA RUTA DESPUÉS DE USARLA!!!
@app.route("/api/limpiar_base_de_datos_ahora/<string:clave_secreta>", methods=["POST"])
def limpiar_base_de_datos(clave_secreta):
    # Usamos una clave secreta simple para evitar que cualquiera la llame
    if clave_secreta != "MI_CLAVE_SUPER_SECRETA_123":
        return jsonify({"error": "No autorizado"}), 403

    conn = conectar_db()
    if conn is None:
        return jsonify({"error": "No se pudo conectar a la base de datos"}), 500

    cursor = conn.cursor()
    query = "TRUNCATE TABLE mediciones;"
    try:
        print(f"LIMPIANDO BASE DE DATOS: Ejecutando query '{query}'")
        cursor.execute(query)
        conn.commit()
        mensaje = "✅ Tabla 'mediciones' vaciada exitosamente."
        print(mensaje)
        return jsonify({"mensaje": mensaje}), 200
    except mysql.connector.Error as err:
        mensaje_error = f"❌ Error al vaciar la tabla: {err}"
        print(mensaje_error)
        conn.rollback()
        return jsonify({"error": mensaje_error}), 500
    finally:
        if conn.is_connected():
            cursor.close()
            conn.close()
            print("Conexión a MySQL cerrada (limpiar_base_de_datos).")
