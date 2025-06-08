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

app = Flask(__name__)

# --- Carga de Modelos de Machine Learning ---
MODEL_SYS_PATH = "modelo_sys.pkl"
MODEL_DIA_PATH = "modelo_dia.pkl"
modelo_sys = None
modelo_dia = None
try:
    if os.path.exists(MODEL_SYS_PATH): modelo_sys = joblib.load(MODEL_SYS_PATH)
    if os.path.exists(MODEL_DIA_PATH): modelo_dia = joblib.load(MODEL_DIA_PATH)
    if modelo_sys and modelo_dia: print("✅ Modelos de ML cargados.")
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

# --- Funciones Auxiliares ---
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

def filtrar_senal_ppg(senal, lowcut=0.5, highcut=5.0, fs=50.0, order=4):
    try:
        nyquist = 0.5 * fs; low = lowcut / nyquist; high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band'); return filtfilt(b, a, senal)
    except Exception: return np.array(senal)

def clasificar_nivel_presion(pas_valor, pad_valor):
    if pas_valor is None or pd.isna(pas_valor): return "---"
    if pas_valor > 180 or pad_valor > 120: return "HT Crisis"
    elif pas_valor >= 140 or pad_valor >= 90: return "HT2"        
    elif (pas_valor >= 130 and pas_valor <= 139) or (pad_valor >= 80 and pad_valor <= 89): return "HT1"        
    elif (pas_valor >= 120 and pas_valor <= 129) and pad_valor < 80: return "Elevada"
    elif pas_valor < 120 and pad_valor < 80: return "Normal"
    else: return "Revisar" 

# Las demás funciones (conectar_db, guardar_medicion_mysql, get_google_drive_service, subir_archivo_a_drive)
# y los demás endpoints (/ultimas_mediciones, /autorizacion, etc.) se mantienen como en la versión anterior.
# Lo importante es el cambio en /api/presion.
# ... (incluir todas las demás funciones y endpoints aquí) ...
# Por brevedad, me centraré en el endpoint modificado que es la clave.

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global autorizado, ultima_estimacion, modelo_sys, modelo_dia, ventana_ir, ventana_red, capturando_entrenamiento, buffer_datos_entrenamiento
    if not modelo_sys or not modelo_dia: return jsonify({"error": "Modelos ML no cargados"}), 500
    
    try:
        data = request.get_json();
        if not data: return jsonify({"error": "Request JSON"}), 400
        
        hr = int(data.get("hr", 0)); ir_val = int(data.get("ir", 0)); red_val = int(data.get("red", 0))
        id_paciente_in = int(data.get("id_paciente", 1)) 

        if capturando_entrenamiento:
            buffer_datos_entrenamiento.append({"hr": hr, "ir": ir_val, "red": red_val, "timestamp": datetime.now()})

        # --- LÓGICA DE CÁLCULO RESTAURADA A TU VERSIÓN ORIGINAL ---
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
                if dc_ir > 0 and dc_red > 0: # Chequeo original
                    ratio = (ac_red / dc_red) / (ac_ir / dc_ir) if ac_ir > 0 and ac_red > 0 else 0
                    spo2_calc = 110 - 25 * ratio
                    spo2_estimada = max(70.0, min(100.0, spo2_calc))
            except Exception as e_spo2: 
                print(f"Error en cálculo SpO2: {e_spo2}"); spo2_estimada = 0.0

        # Predecir SIEMPRE, como en tu código original, incluso si spo2 es 0
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

        # Guardado en CSV para Google Drive (solo si autorizado)
        if autorizado:
            # ... (Lógica de guardado del CSV cuando se implemente el guardado de muestra) ...
            # Por ahora, esta parte se manejará con el endpoint /api/guardar_muestra_entrenamiento
            pass

        # Lógica de guardado en Railway (tu lógica original)
        if ir_val > 20000 and red_val > 15000: 
            guardar_medicion_mysql(id_paciente_in, sys_estimada, dia_estimada, nivel_presion)
        else: 
            print("DEBUG: Condición para guardar en DB no cumplida (según umbrales IR/RED origen).")

        return jsonify({
            "sys": sys_estimada, "dia": dia_estimada, 
            "spo2": spo2_estimada, "nivel": nivel_presion
        }), 200
    except Exception as e: 
        print(f"❌ Error en /api/presion: {e}"); import traceback; traceback.print_exc()
        return jsonify({"error": "Error interno", "detalle": str(e)}), 500
        
# AÑADE AQUÍ EL RESTO DE TUS ENDPOINTS:
# /
# /api/ultimas_mediciones
# /api/autorizacion
# /api/iniciar_captura_entrenamiento
# /api/detener_captura_entrenamiento
# /api/guardar_muestra_entrenamiento

# Y TAMBIÉN EL BLOQUE if __name__ == "__main__":
# if __name__ == "__main__":
#     port = int(os.environ.get("PORT", 10000)) 
#     app.run(host='0.0.0.0', port=port, debug=True)

```

*He dejado placeholders para el resto de funciones y endpoints para centrarme en la corrección del endpoint `/api/presion`. Deberías copiar esta función `api_procesar_presion` y reemplazar la que tienes en tu archivo `app.py` completo.*

### **Paso 2: Código `index.html` Actualizado y SIN Valores Crudos**

Este es el `index.html` con el diseño mejorado y la lógica de captura flexible que pediste, pero **sin mostrar los valores crudos de IR y RED**, y con el campo para el pulso de referencia.


```html
<!DOCTYPE html>
<html lang="es">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Panel de Control - Monitor de PA</title>
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
        :root {
            --primary-color: #3f51b5; --secondary-color: #f50057; --success-color: #4CAF50;
            --warning-color: #ff9800; --light-bg: #f4f7f6; --card-bg: #ffffff;
            --text-color: #333; --text-light: #666; --border-color: #e0e0e0;
            --shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        }
        body {
            font-family: 'Roboto', sans-serif; margin: 0; padding: 20px;
            background-color: var(--light-bg); color: var(--text-color);
            display: flex; flex-direction: column; align-items: center; gap: 20px;
        }
        .main-title { font-size: 2.5em; color: var(--primary-color); font-weight: 700; }
        .container {
            background-color: var(--card-bg); padding: 25px; border-radius: 12px;
            box-shadow: var(--shadow); width: 95%; max-width: 800px; text-align: center;
        }
        h2 {
            font-size: 1.5em; color: var(--primary-color); margin-top: 0;
            padding-bottom: 10px; border-bottom: 2px solid var(--border-color);
        }
        .status-display {
            margin: 15px auto; padding: 12px; border-radius: 8px; font-weight: 500;
            color: white; font-size: 1.1em; max-width: 400px;
        }
        .status-autorizado { background-color: var(--success-color); }
        .status-detenido { background-color: var(--secondary-color); }
        .status-capturando { background-color: var(--warning-color); }
        .data-grid {
            display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            gap: 15px; text-align: left; margin-top: 20px;
        }
        .data-item {
            background: var(--light-bg); padding: 15px; border-radius: 8px;
            border-left: 5px solid var(--primary-color);
        }
        .data-item strong { display: block; color: var(--text-light); margin-bottom: 5px; font-size: 0.9em; text-transform: uppercase; }
        .data-item span { font-size: 1.6em; font-weight: 700; color: var(--primary-color); }
        #nivelVal { font-weight: 700; }
        button {
            padding: 12px 24px; font-size: 1em; font-weight: 500; color: white;
            border: none; border-radius: 8px; cursor: pointer;
            transition: background-color 0.3s, transform 0.1s; margin: 10px 5px;
        }
        button:disabled { background-color: #ccc; cursor: not-allowed; }
        button:active:not(:disabled) { transform: scale(0.97); }
        .btn-toggle-off { background-color: var(--success-color); }
        .btn-toggle-on { background-color: var(--secondary-color); }
        .btn-accion { background-color: #2196F3; }
        .btn-guardar { background-color: #FFC107; color: #333;}
        .form-group {
            margin: 20px 0; display: flex; flex-wrap: wrap; justify-content: center;
            align-items: center; gap: 15px;
        }
        .form-group label { font-weight: 500; }
        input[type="number"] {
            padding: 10px; border: 1px solid var(--border-color); border-radius: 6px;
            width: 80px; font-size: 1.1em; text-align: center;
        }
        table { width: 100%; margin-top: 15px; border-collapse: collapse; }
        th, td { border: 1px solid var(--border-color); padding: 12px; text-align: left; }
        th { background-color: var(--primary-color); color: white; font-weight: 500; }
        tr:nth-child(even) { background-color: var(--light-bg); }
        .message-area { font-weight: 500; margin-top: 15px; padding: 10px; border-radius: 6px; display: none; }
        .message-error { color: #721c24; background-color: #f8d7da; }
        .message-info { color: #155724; background-color: #d4edda; }
    </style>
</head>
<body>

    <h1 class="main-title">Panel de Control del Monitor de PA</h1>

    <div class="container">
        <h2>Estado y Controles</h2>
        <div id="auth-status-display" class="status-display">Cargando...</div>
        <button id="toggleAuthButton">Cargando...</button>
        <p id="message-area" class="message-area"></p>
        
        <div class="training-controls" id="training-capture-section" style="display: none;">
             <h2>Captura para Muestra de Entrenamiento ML</h2>
             <div id="captura-status-display" class="status-display">Captura Inactiva</div>
             <button id="toggleTrainingCaptureButton">Iniciar Captura</button>
             <div class="form-group">
                <div>
                    <label for="pasRef">PAS Ref:</label>
                    <input type="number" id="pasRef" placeholder="120">
                </div>
                <div>
                    <label for="padRef">PAD Ref:</label>
                    <input type="number" id="padRef" placeholder="80">
                </div>
                <div>
                    <label for="hrRef">Pulso Ref:</label>
                    <input type="number" id="hrRef" placeholder="75">
                </div>
             </div>
             <button id="saveTrainingSampleButton" disabled>Guardar Muestra de Entrenamiento</button>
             <p style="font-size: 0.9em; color: var(--text-light);"><strong>Instrucciones:</strong> 1. Inicie la captura. 2. Tome la medición. 3. Detenga la captura. 4. Ingrese los 3 valores de referencia. 5. Guarde.</p>
        </div>
    </div>

    <div class="container">
        <h2>Última Estimación Recibida (Tiempo Real)</h2>
        <div class="data-grid">
            <div class="data-item"><Strong>PAS (Sistólica)</Strong> <span id="sysVal">---</span> mmHg</div>
            <div class="data-item"><Strong>PAD (Diastólica)</Strong> <span id="diaVal">---</span> mmHg</div>
            <div class="data-item"><Strong>SpO2</Strong> <span id="spo2Val">---</span> %</div>
            <div class="data-item"><Strong>HR</Strong> <span id="hrVal">---</span> bpm</div>
            <div class="data-item"><Strong>Nivel</Strong> <span id="nivelVal">---</span></div>
            <div class="data-item"><Strong>Timestamp (UTC)</Strong> <span id="timestampVal">---</span></div>
        </div>
    </div>

    <div class="container">
        <h2>Últimas 20 Mediciones Guardadas (Railway)</h2>
        <p id="history-error-message" class="message-area error-message"></p>
        <table id="medicionesTable">
            <thead>
                <tr>
                    <th>ID</th>
                    <th>ID Paciente</th>
                    <th>PAS</th>
                    <th>PAD</th>
                    <th>Nivel</th>
                </tr>
            </thead>
            <tbody>
                <tr><td colspan="5" style="text-align:center;">Cargando...</td></tr>
            </tbody>
        </table>
    </div>

    <script>
        let estadoAutorizadoGeneral = false;
        let estadoCapturandoEntrenamiento = false;

        const ui = {
            authButton: document.getElementById('toggleAuthButton'),
            authStatus: document.getElementById('auth-status-display'),
            messageArea: document.getElementById('message-area'),
            trainingSection: document.getElementById('training-capture-section'),
            toggleCaptureButton: document.getElementById('toggleTrainingCaptureButton'),
            saveSampleButton: document.getElementById('saveTrainingSampleButton'),
            pasRefInput: document.getElementById('pasRef'),
            padRefInput: document.getElementById('padRef'),
            hrRefInput: document.getElementById('hrRef'), 
            captureStatus: document.getElementById('captura-status-display'),
            sysVal: document.getElementById('sysVal'),
            diaVal: document.getElementById('diaVal'),
            spo2Val: document.getElementById('spo2Val'),
            hrVal: document.getElementById('hrVal'),
            nivelVal: document.getElementById('nivelVal'),
            timestampVal: document.getElementById('timestampVal'),
            historyError: document.getElementById('history-error-message'),
            medicionesTbody: document.getElementById('medicionesTable').getElementsByTagName('tbody')[0]
        };

        function mostrarMensaje(texto, tipo = 'info', duracion = 4000) {
            ui.messageArea.textContent = texto;
            ui.messageArea.className = `message-area message-${tipo}`;
            ui.messageArea.style.display = 'block';
            setTimeout(() => ui.messageArea.style.display = 'none', duracion);
        }

        function actualizarUI() {
            if (estadoAutorizadoGeneral) {
                ui.authButton.textContent = 'Detener Registro General';
                ui.authButton.className = 'btn-toggle-on';
                ui.authStatus.textContent = 'Modo General: Registro Autorizado';
                ui.authStatus.className = 'status-display status-autorizado';
                ui.trainingSection.style.display = 'block';
            } else {
                ui.authButton.textContent = 'Autorizar Registro General';
                ui.authButton.className = 'btn-toggle-off';
                ui.authStatus.textContent = 'Modo General: Registro Detenido';
                ui.authStatus.className = 'status-display status-detenido';
                ui.trainingSection.style.display = 'none';
            }

            if (estadoCapturandoEntrenamiento) {
                ui.captureStatus.textContent = 'Captura para Entrenamiento: ACTIVA';
                ui.captureStatus.className = 'status-display status-capturando';
                ui.toggleCaptureButton.textContent = "Detener Captura";
                ui.toggleCaptureButton.className = 'btn-toggle-on';
                ui.saveSampleButton.disabled = true;
            } else {
                ui.captureStatus.textContent = 'Captura para Entrenamiento: Inactiva';
                ui.captureStatus.className = 'status-display status-detenido';
                ui.toggleCaptureButton.textContent = "Iniciar Captura de Segmento";
                ui.toggleCaptureButton.className = 'btn-accion';
                ui.toggleCaptureButton.disabled = !estadoAutorizadoGeneral;
                ui.saveSampleButton.disabled = false;
            }
        }

        function toggleAutorizacion() {
            fetch('/api/autorizacion', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ autorizado: !estadoAutorizadoGeneral }),
            })
            .then(response => response.json())
            .then(data => {
                if (data.autorizado !== undefined) {
                    estadoAutorizadoGeneral = data.autorizado;
                    if (!estadoAutorizadoGeneral && estadoCapturandoEntrenamiento) {
                        fetch('/api/detener_captura_entrenamiento', { method: 'POST' });
                        estadoCapturandoEntrenamiento = false;
                    }
                    actualizarUI();
                    mostrarMensaje(data.mensaje || `Estado general cambiado.`);
                } else { mostrarMensaje('Error en respuesta del servidor.', 'error'); }
            })
            .catch((error) => { console.error('Error:', error); mostrarMensaje('Error de red.', 'error'); });
        }

        function toggleCapturaEntrenamiento() {
            const endpoint = estadoCapturandoEntrenamiento ? '/api/detener_captura_entrenamiento' : '/api/iniciar_captura_entrenamiento';
            fetch(endpoint, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if(data.capturando !== undefined) {
                    estadoCapturandoEntrenamiento = data.capturando;
                    actualizarUI();
                    mostrarMensaje(data.mensaje || "Estado de captura cambiado.");
                } else { mostrarMensaje(data.error || "Error al cambiar estado.", "error");}
            })
            .catch(error => { console.error('Error:', error); mostrarMensaje("Error de red.", "error");});
        }

        function guardarMuestraEntrenamiento() {
            const pas = ui.pasRefInput.value;
            const pad = ui.padRefInput.value;
            const hr = ui.hrRefInput.value; 

            if (!pas || !pad || !hr) {
                mostrarMensaje("Por favor, ingrese PAS, PAD y Pulso de referencia.", "error"); return;
            }
            if (estadoCapturandoEntrenamiento) {
                mostrarMensaje("Por favor, primero detenga la captura antes de guardar.", "error"); return;
            }

            fetch('/api/guardar_muestra_entrenamiento', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ 
                    pas_referencia: parseFloat(pas), 
                    pad_referencia: parseFloat(pad),
                    hr_referencia: parseFloat(hr) 
                }),
            })
            .then(response => response.json())
            .then(data => {
                mostrarMensaje(data.mensaje || data.error, data.error ? 'error' : 'info');
                if (!data.error) {
                    ui.pasRefInput.value = ''; ui.padRefInput.value = ''; ui.hrRefInput.value = ''; 
                    actualizarUI();
                }
            })
            .catch(error => { console.error('Error:', error); mostrarMensaje("Error de red al guardar.", "error");});
        }

        function actualizarUltimaEstimacion() {
            fetch('/api/ultima_estimacion').then(response => response.ok ? response.json() : Promise.reject('Respuesta no OK'))
            .then(data => {
                ui.sysVal.textContent = data.sys || '---';
                ui.diaVal.textContent = data.dia || '---';
                ui.spo2Val.textContent = data.spo2 || '---';
                ui.hrVal.textContent = data.hr || '---';
                ui.nivelVal.textContent = data.nivel || '---';
                ui.timestampVal.textContent = data.timestamp || '---';
                if (data.modo_autorizado !== undefined && data.modo_autorizado !== estadoAutorizadoGeneral) {
                    estadoAutorizadoGeneral = data.modo_autorizado;
                    actualizarUI();
                }
                if (data.capturando_entrenamiento !== undefined && data.capturando_entrenamiento !== estadoCapturandoEntrenamiento) {
                    estadoCapturandoEntrenamiento = data.capturando_entrenamiento;
                    actualizarUI();
                }
            }).catch(error => console.error('Error al actualizar última estimación:', error));
        }

        function actualizarTablaMediciones() {
            ui.historyError.style.display = 'none';
            fetch('/api/ultimas_mediciones').then(response => {
                if (!response.ok) { throw new Error(`Error HTTP ${response.status}`); } return response.json();
            }).then(data => {
                ui.medicionesTbody.innerHTML = ''; 
                if (data && Array.isArray(data) && data.length > 0) {
                    data.forEach(medicion => {
                        let row = ui.medicionesTbody.insertRow();
                        row.insertCell().textContent = medicion.id ?? 'N/A';
                        row.insertCell().textContent = medicion.id_paciente ?? 'N/A';
                        row.insertCell().textContent = medicion.sys ?? 'N/A';
                        row.insertCell().textContent = medicion.dia ?? 'N/A';
                        row.insertCell().textContent = medicion.nivel || 'N/A';
                    });
                } else {
                    let row = ui.medicionesTbody.insertRow(); let cell = row.insertCell(); cell.colSpan = 5; 
                    cell.textContent = 'No hay mediciones guardadas.'; cell.style.textAlign = 'center';
                }
            }).catch(error => {
                console.error('Error al actualizar tabla:', error);
                ui.historyError.textContent = `Error al cargar historial: ${error.message}`;
                ui.historyError.style.display = 'block';
            });
        }

        document.addEventListener('DOMContentLoaded', function() {
            ui.authButton.addEventListener('click', toggleAutorizacion);
            ui.toggleCaptureButton.addEventListener('click', toggleCapturaEntrenamiento);
            ui.saveSampleButton.addEventListener('click', guardarMuestraEntrenamiento);

            fetch('/api/autorizacion')
                .then(response => response.json())
                .then(data => {
                    if (data.autorizado !== undefined) estadoAutorizadoGeneral = data.autorizado;
                    if (data.capturando_entrenamiento !== undefined) estadoCapturandoEntrenamiento = data.capturando_entrenamiento;
                    actualizarUI();
                })
                .catch(error => {
                    console.error('Error al obtener estado inicial:', error)
                    mostrarMensaje('No se pudo cargar el estado inicial del servidor.', 'error');
                });
            
            actualizarUltimaEstimacion(); 
            actualizarTablaMediciones(); 

            setInterval(actualizarUltimaEstimacion, 5000); 
            setInterval(actualizarTablaMediciones, 30000); 
        });
    </script>

</body>
</html>
