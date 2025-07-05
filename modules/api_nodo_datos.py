# modules/api_nodo_datos.py
# API especializada para recibir y procesar datos del ESP32
# ACTUALIZADO: Sistema de 100 muestras para predicciones ML precisas

from flask import Blueprint, request, jsonify
import time
import numpy as np
from datetime import datetime
import logging

# Crear blueprint para la API de nodos
api_nodo_bp = Blueprint('api_nodo', __name__)

# Logger específico para este módulo
logger = logging.getLogger(__name__)

class SampleBuffer:
    """Buffer para acumular 100 muestras por paciente antes de predicción ML"""
    
    def __init__(self):
        self.patient_buffers = {}  # {patient_id: [samples]}
        self.target_samples = 100
        self.max_inactive_time = 300  # 5 minutos para limpiar buffers inactivos
    
    def add_sample(self, patient_id, ir, red, timestamp):
        """Añadir muestra al buffer del paciente"""
        if patient_id not in self.patient_buffers:
            self.patient_buffers[patient_id] = []
        
        sample = {
            'ir': float(ir),
            'red': float(red),
            'timestamp': timestamp,
            'time_added': time.time()
        }
        
        self.patient_buffers[patient_id].append(sample)
        
        # Mantener solo las últimas 100 muestras
        if len(self.patient_buffers[patient_id]) > self.target_samples:
            self.patient_buffers[patient_id] = self.patient_buffers[patient_id][-self.target_samples:]
        
        return len(self.patient_buffers[patient_id])
    
    def get_sample_count(self, patient_id):
        """Obtener número de muestras acumuladas del paciente"""
        return len(self.patient_buffers.get(patient_id, []))
    
    def has_enough_samples(self, patient_id):
        """Verificar si tiene las 100 muestras necesarias para predicción"""
        return self.get_sample_count(patient_id) >= self.target_samples
    
    def get_samples_for_ml(self, patient_id):
        """Procesar las 100 muestras y extraer características para ML"""
        if not self.has_enough_samples(patient_id):
            return None
        
        # Obtener las últimas 100 muestras
        samples = self.patient_buffers[patient_id][-self.target_samples:]
        
        # Extraer valores IR y RED
        ir_values = [s['ir'] for s in samples]
        red_values = [s['red'] for s in samples]
        
        # Calcular características necesarias para ML
        processed_data = {
            'hr_calculated': self._calculate_heart_rate(ir_values),
            'spo2_calculated': self._calculate_spo2(ir_values, red_values),
            'ir_mean': np.mean(ir_values),
            'red_mean': np.mean(red_values),
            'ir_std': np.std(ir_values),
            'red_std': np.std(red_values)
        }
        
        return processed_data
    
    def _calculate_heart_rate(self, ir_values):
        """Calcular frecuencia cardíaca usando análisis mejorado de picos en señal IR"""
        try:
            ir_array = np.array(ir_values)
            
            # Suavizar la señal para reducir ruido
            window_size = min(5, len(ir_array) // 10)
            if window_size >= 3:
                # Aplicar filtro de media móvil
                smoothed = np.convolve(ir_array, np.ones(window_size)/window_size, mode='valid')
            else:
                smoothed = ir_array
            
            # Encontrar picos usando método mejorado
            peaks = []
            threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
            
            for i in range(1, len(smoothed) - 1):
                # Un pico debe ser mayor que sus vecinos Y superar el umbral
                if (smoothed[i] > smoothed[i-1] and 
                    smoothed[i] > smoothed[i+1] and 
                    smoothed[i] > threshold):
                    peaks.append(i)
            
            # Filtrar picos demasiado cercanos (evitar doble detección)
            if len(peaks) > 1:
                filtered_peaks = [peaks[0]]
                min_distance = len(smoothed) // 20  # Mínima distancia entre picos
                
                for peak in peaks[1:]:
                    if peak - filtered_peaks[-1] > min_distance:
                        filtered_peaks.append(peak)
                
                peaks = filtered_peaks
            
            if len(peaks) < 2:
                # Si no detecta picos, usar análisis de periodicidad
                return self._calculate_hr_by_fft(ir_array)
            
            # Calcular intervalos entre picos
            intervals = np.diff(peaks)
            
            # Filtrar intervalos anómalos
            if len(intervals) > 2:
                # Eliminar intervalos que están muy fuera del promedio
                mean_interval = np.mean(intervals)
                std_interval = np.std(intervals)
                filtered_intervals = []
                
                for interval in intervals:
                    if abs(interval - mean_interval) < 2 * std_interval:
                        filtered_intervals.append(interval)
                
                if filtered_intervals:
                    intervals = filtered_intervals
            
            avg_interval = np.mean(intervals)
            
            # Convertir a BPM
            # 100 muestras tomadas en 200 segundos = 0.5 Hz de muestreo
            sample_rate = 0.5  # Hz
            time_per_sample = 1.0 / sample_rate  # 2 segundos por muestra
            
            # Tiempo entre picos en segundos
            time_between_beats = avg_interval * time_per_sample
            
            # Frecuencia cardíaca en BPM
            hr = 60.0 / time_between_beats
            
            # Validar rango fisiológico
            if 40 <= hr <= 200:
                return hr
            else:
                # Si está fuera de rango, intentar método alternativo
                return self._calculate_hr_by_fft(ir_array)
                
        except Exception as e:
            logger.warning(f"Error calculando HR con picos: {e}")
            # Fallback a método FFT
            return self._calculate_hr_by_fft(ir_array)
    
    def _calculate_hr_by_fft(self, ir_values):
        """Método alternativo: calcular HR usando análisis de frecuencia (FFT)"""
        try:
            # Aplicar FFT para encontrar la frecuencia dominante
            from scipy import signal
            
            # Remover componente DC y normalizar
            ir_centered = ir_values - np.mean(ir_values)
            
            # Aplicar ventana para reducir efectos de borde
            windowed = ir_centered * signal.windows.hann(len(ir_centered))
            
            # Calcular FFT
            fft = np.fft.fft(windowed)
            freqs = np.fft.fftfreq(len(windowed), d=2.0)  # 2 segundos por muestra
            
            # Solo frecuencias positivas
            positive_freqs = freqs[:len(freqs)//2]
            positive_fft = np.abs(fft[:len(fft)//2])
            
            # Buscar en rango de frecuencias cardíacas (0.67-3.33 Hz = 40-200 BPM)
            min_freq = 40.0 / 60.0  # 40 BPM en Hz
            max_freq = 200.0 / 60.0  # 200 BPM en Hz
            
            valid_indices = np.where((positive_freqs >= min_freq) & (positive_freqs <= max_freq))[0]
            
            if len(valid_indices) > 0:
                valid_fft = positive_fft[valid_indices]
                valid_freqs = positive_freqs[valid_indices]
                
                # Encontrar el pico de frecuencia
                peak_idx = np.argmax(valid_fft)
                dominant_freq = valid_freqs[peak_idx]
                
                # Convertir a BPM
                hr = dominant_freq * 60.0
                
                return hr if 40 <= hr <= 200 else 75  # Default si está fuera de rango
            else:
                return 75  # Valor por defecto si no encuentra frecuencias válidas
                
        except Exception as e:
            logger.warning(f"Error en cálculo FFT: {e}")
            # Usar estimación basada en variabilidad de la señal
            return self._estimate_hr_from_variability(ir_values)
    
    def _estimate_hr_from_variability(self, ir_values):
        """Método de último recurso: estimar HR desde variabilidad de señal"""
        try:
            # Calcular la variabilidad de la señal
            diff = np.diff(ir_values)
            variability = np.std(diff)
            
            # Heurística: mayor variabilidad suele corresponder a mayor HR
            # Esta es una aproximación muy básica
            if variability > 1000:
                return 80  # HR alta
            elif variability > 500:
                return 70  # HR media
            else:
                return 60  # HR baja
                
        except:
            return 75  # Valor por defecto seguro
    
    def _calculate_spo2(self, ir_values, red_values):
        """Calcular SpO2 usando ratio R/IR estándar"""
        try:
            ir_array = np.array(ir_values)
            red_array = np.array(red_values)
            
            # Calcular componentes AC (variabilidad) y DC (promedio)
            ir_ac = np.std(ir_array)
            ir_dc = np.mean(ir_array)
            red_ac = np.std(red_array)
            red_dc = np.mean(red_array)
            
            # Evitar división por cero
            if ir_dc == 0 or red_dc == 0 or ir_ac == 0:
                return 0
            
            # Calcular ratio R (fórmula estándar oximetría)
            r = (red_ac / red_dc) / (ir_ac / ir_dc)
            
            # Ecuación de calibración para MAX30102
            spo2 = 104 - 17 * r
            
            # Validar rango de SpO2
            return spo2 if 70 <= spo2 <= 100 else 0
            
        except Exception as e:
            logger.warning(f"Error calculando SpO2: {e}")
            return 0
    
    def clear_patient_buffer(self, patient_id):
        """Limpiar completamente el buffer de un paciente"""
        if patient_id in self.patient_buffers:
            del self.patient_buffers[patient_id]
            logger.info(f"Buffer limpiado para paciente {patient_id}")
    
    def cleanup_old_samples(self):
        """Limpiar buffers de pacientes inactivos por más de 5 minutos"""
        current_time = time.time()
        inactive_patients = []
        
        for patient_id, samples in self.patient_buffers.items():
            if samples:
                last_sample_time = samples[-1]['time_added']
                if current_time - last_sample_time > self.max_inactive_time:
                    inactive_patients.append(patient_id)
        
        # Limpiar pacientes inactivos
        for patient_id in inactive_patients:
            del self.patient_buffers[patient_id]
            logger.info(f"Buffer eliminado por inactividad: paciente {patient_id}")
    
    def get_buffer_status(self):
        """Obtener estado actual de todos los buffers"""
        status = {}
        for patient_id, samples in self.patient_buffers.items():
            status[patient_id] = {
                'sample_count': len(samples),
                'last_activity': samples[-1]['time_added'] if samples else 0,
                'ready_for_ml': len(samples) >= self.target_samples
            }
        return status

# Crear instancia global del buffer de muestras
sample_buffer = SampleBuffer()

def classify_pressure_level(sys_pressure, dia_pressure):
    """Clasificar nivel de presión arterial según guías médicas AHA/ESC"""
    if sys_pressure == 0 or dia_pressure == 0:
        return "Sin datos"
    
    try:
        sys_val, dia_val = float(sys_pressure), float(dia_pressure)
    except (ValueError, TypeError):
        return "Error"
    
    # Crisis de Hipertensión (SYS > 180 O DIA > 120)
    if sys_val > 180 or dia_val > 120:
        return "HT Crisis"
    
    # Hipertensión Etapa 2 (SYS ≥ 140 O DIA ≥ 90)
    if sys_val >= 140 or dia_val >= 90:
        return "HT2"
    
    # Hipertensión Etapa 1 (SYS 130-139 O DIA 80-89)
    if (130 <= sys_val <= 139) or (80 <= dia_val <= 89):
        return "HT1"
    
    # Elevada (SYS 120-129 Y DIA < 80)
    if 120 <= sys_val <= 129 and dia_val < 80:
        return "Elevada"
    
    # Normal (SYS < 120 Y DIA < 80)
    if sys_val < 120 and dia_val < 80:
        return "Normal"
    
    # Caso edge: valores que no encajan perfectamente
    return "Revisar"

@api_nodo_bp.route('/raw_data', methods=['POST'])
def receive_raw_data():
    """
    Endpoint principal para recibir datos del ESP32
    NUEVO: Acumula 100 muestras antes de hacer predicción ML
    """
    try:
        data = request.get_json()
        if not data:
            logger.warning("Datos JSON vacíos recibidos")
            return jsonify({"error": "No JSON data"}), 400

        # Extraer datos del ESP32
        patient_id = data.get('id_paciente', 1)
        ir_value = float(data.get('ir', 0))
        red_value = float(data.get('red', 0))
        timestamp = data.get('timestamp', int(time.time() * 1000))

        logger.info(f"Datos ESP32 - Paciente:{patient_id} IR:{ir_value} RED:{red_value}")

        # CASO ESPECIAL: ID 999 para prueba de buzzer
        if patient_id == 999:
            logger.info("Modo prueba buzzer activado")
            response = {
                "sys": 185, "dia": 125, "hr": 99, "spo2": 99, 
                "nivel": "HT Crisis", "muestras_recolectadas": 100
            }
            if hasattr(api_nodo_bp, 'websocket_handler'):
                api_nodo_bp.websocket_handler.emit_update(data)
            return jsonify(response)

        # Añadir muestra al buffer del paciente
        sample_count = sample_buffer.add_sample(patient_id, ir_value, red_value, timestamp)
        
        logger.info(f"Muestra añadida: {sample_count}/100 para paciente {patient_id}")

        # Respuesta por defecto mientras se recolectan muestras - SIN CALCULOS
        response = {
            "sys": 0, 
            "dia": 0, 
            "hr": 0, 
            "spo2": 0,
            "nivel": f"Recolectando datos... {sample_count}/100",
            "muestras_recolectadas": sample_count,
            "muestras_necesarias": 100,
            "calidad": "collecting"
        }

        # SOLO calcular cuando se tengan EXACTAMENTE 100 muestras
        if sample_count == 100:
            logger.info(f"EXACTAMENTE 100 muestras alcanzadas para paciente {patient_id} - Iniciando cálculos ML")
            
            # Obtener datos procesados para ML
            ml_data = sample_buffer.get_samples_for_ml(patient_id)
            
            if ml_data and hasattr(api_nodo_bp, 'ml_processor') and api_nodo_bp.ml_processor.is_ready():
                try:
                    # Realizar predicción ML SOLO con las 100 muestras completas
                    sys_pred, dia_pred = api_nodo_bp.ml_processor.predict_pressure(
                        ml_data['hr_calculated'],
                        ml_data['spo2_calculated'],
                        ml_data['ir_mean'],
                        ml_data['red_mean'],
                        ml_data['ir_std'],
                        ml_data['red_std']
                    )
                    
                    # Actualizar respuesta con resultados ML
                    response.update({
                        "sys": round(sys_pred, 1),
                        "dia": round(dia_pred, 1),
                        "hr": round(ml_data['hr_calculated'], 0),
                        "spo2": round(ml_data['spo2_calculated'], 0),
                        "nivel": classify_pressure_level(sys_pred, dia_pred),
                        "calidad": "complete"
                    })
                    
                    logger.info(f"Predicción ML completada - SYS:{sys_pred} DIA:{dia_pred} HR:{ml_data['hr_calculated']} SpO2:{ml_data['spo2_calculated']}")
                    
                    # Guardar medición en base de datos - CORREGIDO
                    if hasattr(api_nodo_bp, 'db_manager') and api_nodo_bp.db_manager.is_connected():
                        measurement_data = {
                            'id_paciente': patient_id,
                            'sys': sys_pred,                     
                            'dia': dia_pred,                     
                            'hr_ml': ml_data['hr_calculated'],
                            'spo2_ml': ml_data['spo2_calculated'],
                            'nivel': response["nivel"]           
                        }
                        api_nodo_bp.db_manager.save_measurement_async(measurement_data)
                        logger.info(f"Medición guardada en BD: SYS={sys_pred} DIA={dia_pred}")
                    
                    # Verificar y enviar alertas si es necesario
                    if hasattr(api_nodo_bp, 'alert_system'):
                        alert_data = {
                            'patient_id': patient_id,
                            'nivel': response["nivel"],
                            'sys': sys_pred,
                            'dia': dia_pred,
                            'hr': ml_data['hr_calculated'],
                            'spo2': ml_data['spo2_calculated']
                        }
                        api_nodo_bp.alert_system.check_and_send_alert(alert_data)
                    
                    # Notificar nuevos datos guardados vía WebSocket
                    if hasattr(api_nodo_bp, 'websocket_handler'):
                        api_nodo_bp.websocket_handler.emit_new_record_saved()
                    
                    # Limpiar buffer después de predicción exitosa para reiniciar ciclo
                    sample_buffer.clear_patient_buffer(patient_id)
                    logger.info(f"Buffer limpiado para paciente {patient_id} - Listo para nuevo ciclo")
                    
                except Exception as e:
                    logger.error(f"Error en predicción ML: {e}")
                    response["nivel"] = "Error ML"
                    response["calidad"] = "error"
            else:
                logger.warning("ML processor no disponible o no configurado")
                response["nivel"] = "ML no disponible"
                response["calidad"] = "error"
        
        # Limpiar buffers antiguos periódicamente
        sample_buffer.cleanup_old_samples()
        
        # Notificar actualización vía WebSocket
        if hasattr(api_nodo_bp, 'websocket_handler'):
            update_data = {**data, **response}
            api_nodo_bp.websocket_handler.emit_update(update_data)

        logger.info(f"Respuesta enviada - Muestras:{sample_count}/100 Nivel:{response['nivel']}")
        return jsonify(response)

    except Exception as e:
        logger.error(f"Error procesando datos del ESP32: {e}")
        return jsonify({
            "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
            "nivel": "Error servidor", 
            "muestras_recolectadas": 0,
            "calidad": "error"
        }), 500

@api_nodo_bp.route('/patient_status/<int:patient_id>', methods=['GET'])
def get_patient_status(patient_id):
    """Obtener estado actual del buffer de un paciente específico"""
    try:
        sample_count = sample_buffer.get_sample_count(patient_id)
        has_enough = sample_buffer.has_enough_samples(patient_id)
        
        status = {
            "patient_id": patient_id,
            "sample_count": sample_count,
            "samples_needed": 100,
            "ready_for_ml": has_enough,
            "progress_percentage": (sample_count / 100) * 100,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(status)
        
    except Exception as e:
        logger.error(f"Error obteniendo estado del paciente {patient_id}: {e}")
        return jsonify({"error": "Error interno"}), 500

@api_nodo_bp.route('/buffer_status', methods=['GET'])
def get_buffer_status():
    """Obtener estado completo de todos los buffers de pacientes"""
    try:
        status = sample_buffer.get_buffer_status()
        return jsonify({
            "active_patients": len(status),
            "patient_buffers": status,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error obteniendo estado de buffers: {e}")
        return jsonify({"error": "Error interno"}), 500

@api_nodo_bp.route('/reset_buffer/<int:patient_id>', methods=['POST'])
def reset_patient_buffer(patient_id):
    """Resetear buffer de muestras de un paciente específico"""
    try:
        sample_buffer.clear_patient_buffer(patient_id)
        logger.info(f"Buffer manual reset para paciente {patient_id}")
        
        return jsonify({
            "status": "success",
            "message": f"Buffer del paciente {patient_id} reseteado",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error reseteando buffer: {e}")
        return jsonify({"error": "Error interno"}), 500

@api_nodo_bp.route('/system_metrics', methods=['GET'])
def get_system_metrics():
    """Obtener métricas del sistema de procesamiento de datos"""
    try:
        buffer_status = sample_buffer.get_buffer_status()
        
        metrics = {
            "active_patients": len(buffer_status),
            "total_samples": sum(status['sample_count'] for status in buffer_status.values()),
            "patients_ready_for_ml": sum(1 for status in buffer_status.values() if status['ready_for_ml']),
            "buffer_details": buffer_status,
            "target_samples": 100,
            "timestamp": datetime.now().isoformat()
        }
        
        return jsonify(metrics)
    except Exception as e:
        logger.error(f"Error obteniendo métricas del sistema: {e}")
        return jsonify({"error": "Error interno"}), 500

# Manejo de errores específicos
@api_nodo_bp.errorhandler(ValueError)
def handle_value_error(e):
    logger.error(f"Error de valor en API nodo: {e}")
    return jsonify({"error": "Datos inválidos"}), 400

@api_nodo_bp.errorhandler(KeyError)
def handle_key_error(e):
    logger.error(f"Clave faltante en API nodo: {e}")
    return jsonify({"error": "Datos incompletos"}), 400

def init_api_module(app):
    """Inicializar el módulo API con la aplicación Flask"""
    logger.info("Inicializando módulo API de nodos con sistema de 100 muestras")
    
    # Configurar logging específico
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('[API-NODO] %(asctime)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    
    logger.info("Módulo API de nodos inicializado correctamente")

# Inicializar automáticamente cuando se importe
init_api_module(None)
