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
from scipy.signal import butter, filtfilt, find_peaks

# Importar librería para peticiones HTTP
import requests
from urllib.parse import quote 

app = Flask(__name__)

# --- Carga de Modelos y Configuración Global (sin cambios) ---
# ... (mantener como en la versión anterior) ...

# --- NUEVAS FUNCIONES DE PROCESAMIENTO AVANZADO ---
def filtrar_senal_ppg(senal, lowcut=0.5, highcut=4.0, fs=50.0, order=4):
    """Aplica un filtro Butterworth pasabanda a la señal PPG."""
    # fs: Frecuencia de muestreo. ¡DEBE COINCIDIR CON LA DEL SENSOR!
    # El sensor se configuró a 100Hz, pero el ESP32 envía a 5Hz (cada 200ms).
    # Usaremos una fs estimada de 5.0 para el filtro.
    fs_real = 5.0 
    try:
        nyquist = 0.5 * fs_real
        low = lowcut / nyquist
        high = highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        senal_filtrada = filtfilt(b, a, senal)
        return senal_filtrada
    except Exception as e:
        print(f"Error al filtrar señal: {e}")
        return np.array(senal)

def procesar_segmento_ppg(segmento_ir, segmento_red):
    """
    Procesa un segmento de datos PPG para calcular HR y SpO2 de forma robusta.
    Esta función emula la lógica de algoritmos avanzados como el de Maxim.
    """
    fs = 5.0 # Frecuencia de muestreo estimada (5Hz)

    # 1. Filtrar las señales para eliminar ruido
    ir_filtrado = filtrar_senal_ppg(np.array(segmento_ir), fs=fs)
    red_filtrado = filtrar_senal_ppg(np.array(segmento_red), fs=fs)
    
    # 2. Detección de picos en la señal IR (más fuerte) para HR
    # `distance` asegura que los picos estén separados por al menos 0.4 segundos (máx 150 bpm)
    # `height` asegura que el pico sea significativo
    try:
        picos, _ = find_peaks(ir_filtrado, height=np.mean(ir_filtrado), distance=fs*0.4)
        
        if len(picos) >= 2:
            # Calcular el promedio de los intervalos entre picos
            intervalos_entre_picos_ms = np.mean(np.diff(picos)) * (1000.0 / fs)
            hr_calculado = 60000.0 / intervalos_entre_picos_ms
            if hr_calculado < 40 or hr_calculado > 200: hr_calculado = 0.0 # Fuera de rango fisiológico
        else:
            hr_calculado = 0.0 # No se encontraron suficientes picos
    except Exception as e_hr:
        print(f"Error en cálculo de HR: {e_hr}")
        hr_calculado = 0.0

    # 3. Cálculo de SpO2 usando el método AC/DC
    spo2_calculado = 0.0
    try:
        dc_ir = np.mean(ir_filtrado); dc_red = np.mean(red_filtrado)
        
        # AC se calcula como la desviación estándar para robustez
        ac_ir = np.std(ir_filtrado); ac_red = np.std(red_filtrado)

        if dc_ir > 0 and dc_red > 0 and ac_ir > 0 and ac_red > 0:
            ratio = (ac_red / dc_red) / (ac_ir / dc_ir)
            spo2_calc = 104 - 17 * ratio # Otra fórmula empírica común
            spo2_calculado = max(70.0, min(100.0, spo2_calc))
        else:
            spo2_calculado = 0.0
    except Exception as e_spo2:
        print(f"Error en cálculo de SpO2: {e_spo2}")
        spo2_calculado = 0.0
        
    return round(hr_calculado, 1), round(spo2_calculado, 1)

# ... (El resto de tus funciones como clasificar_nivel_presion, conectar_db, etc. se mantienen igual) ...

@app.route("/api/presion", methods=["POST"])
def api_procesar_presion():
    global ventana_ir, ventana_red, ultima_estimacion, modelo_sys, modelo_dia
    
    # ... (lógica para recibir datos ir_val, red_val del ESP32) ...

    # Añadir nuevas lecturas a la ventana deslizante
    ventana_ir.append(ir_val)
    ventana_red.append(red_val)
    if len(ventana_ir) > 50: # Mantener una ventana de 10 segundos (50 muestras a 5Hz)
        ventana_ir.pop(0)
        ventana_red.pop(0)

    hr_estimado_rt = 0.0
    spo2_estimada_rt = 0.0
    
    # Solo procesar si tenemos suficientes datos en la ventana
    if len(ventana_ir) >= 25: # Necesitar al menos 5 segundos de datos (25 muestras)
        hr_estimado_rt, spo2_estimada_rt = procesar_segmento_ppg(ventana_ir, ventana_red)

    # --- El resto del flujo como antes ---
    if spo2_estimada_rt > 0 and hr_estimado_rt > 0:
        entrada_df = pd.DataFrame([[hr_estimado_rt, spo2_estimada_rt]], columns=['hr', 'spo2'])
        sys_estimada = round(modelo_sys.predict(entrada_df)[0], 2)
        dia_estimada = round(modelo_dia.predict(entrada_df)[0], 2)
    else:
        sys_estimada = -1.0; dia_estimada = -1.0
    
    nivel_presion = clasificar_nivel_presion(sys_estimada, dia_estimada)

    # Actualizar `ultima_estimacion`
    ultima_estimacion["hr"] = f"{hr_estimado_rt:.1f}" if hr_estimado_rt > 0 else "---"
    # ... (resto de la actualización de ultima_estimacion) ...

    # Responder al ESP32 con los nuevos valores calculados por el servidor
    return jsonify({ "sys": sys_estimada, "dia": dia_estimada, "spo2": spo2_estimada_rt, "hr": hr_estimado_rt, "nivel": nivel_presion }), 200

