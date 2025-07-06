# modules/api_nodo_datos.py
# API especializada para recibir y procesar datos del ESP32
# CORREGIDO: Sistema de 50 muestras con limpieza automática

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
    """Buffer para acumular 50 muestras por paciente antes de predicción ML"""
    
    def __init__(self):
        self.patient_buffers = {}  # {patient_id: [samples]}
        self.target_samples = 50  # 50 muestras
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
        
        # Mantener solo las últimas 50 muestras
        if len(self.patient_buffers[patient_id]) > self.target_samples:
            self.patient_buffers[patient_id] = self.patient_buffers[patient_id][-self.target_samples:]
        
        return len(self.patient_buffers[patient_id])
    
    def get_sample_count(self, patient_id):
        """Obtener número de muestras acumuladas del paciente"""
        return len(self.patient_buffers.get(patient_id, []))
    
    def has_enough_samples(self, patient_id):
        """Verificar si tiene las 50 muestras necesarias para predicción"""
        return self.get_sample_count(patient_id) >= self.target_samples
    
    def get_samples_for_ml(self, patient_id):
        """Procesar las 50 muestras y extraer características para ML"""
        if not self.has_enough_samples(patient_id):
            return None
        
        # Obtener las últimas 50 muestras
        samples = self.patient_buffers[patient_id][-self.target_samples:]
        
        # Extraer valores IR y RED
        ir_values = [s['ir'] for s in samples]
        red_values = [s['red'] for s in samples]
        
        # Calcular características necesarias para ML
        try:
            processed_data = {
                'hr_calculated': self._calculate_heart_rate(ir_values),
                'spo2_calculated': self._calculate_spo2(ir_values, red_values),
                'ir_mean': np.mean(ir_values),
                'red_mean': np.mean(red_values),
                'ir_std': np.std(ir_values),
                'red_std': np.std(red_values)
            }
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error procesando muestras para ML: {e}")
            return None
    
    def _calculate_heart_rate(self, ir_values):
        """Calcular frecuencia cardíaca usando análisis mejorado de picos en señal IR"""
        try:
            ir_array = np.array(ir_values, dtype=float)
            
            # Validar entrada
            if len(ir_array) < 10:
                return 75.0  # Default
            
            # Suavizar la señal
            window_size = min(5, len(ir_array) // 10)
            if window_size >= 3:
                smoothed = np.convolve(ir_array, np.ones(window_size)/window_size, mode='valid')
            else:
                smoothed = ir_array
            
            # Encontrar picos
            peaks = []
            if len(smoothed) > 2:
                threshold = np.mean(smoothed) + 0.5 * np.std(smoothed)
                
                for i in range(1, len(smoothed) - 1):
                    if (smoothed[i] > smoothed[i-1] and 
                        smoothed[i] > smoothed[i+1] and 
                        smoothed[i] > threshold):
                        peaks.append(i)
            
            # Calcular HR desde picos
            if len(peaks) >= 2:
                intervals = np.diff(peaks)
                if len(intervals) > 0:
                    avg_interval = np.mean(intervals)
                    sample_rate = 0.5  # Hz (50 muestras en 100 segundos)
                    time_per_sample = 1.0 / sample_rate
                    time_between_beats = avg_interval * time_per_sample
                    hr = 60.0 / time_between_beats
                    
                    # Validar rango
                    if 40 <= hr <= 150:
                        return float(hr)
            
            # Método alternativo: usar variabilidad
            return self._estimate_hr_from_variability(ir_values)
            
        except Exception as e:
            logger.warning(f"Error calculando HR: {e}")
            return 75.0
    
    def _estimate_hr_from_variability(self, ir_values):
        """Método alternativo para estimar HR"""
        try:
            # Calcular variabilidad básica
            diff = np.diff(ir_values)
            variability = np.std(diff)
            
            # Mapear variabilidad a HR (heurística)
            if variability > 1500:
                return 85.0
            elif variability > 800:
                return 75.0
            elif variability > 400:
                return 68.0
            else:
                return 62.0
                
        except:
            return 75.0
    
    def _calculate_spo2(self, ir_values, red_values):
        """Calcular SpO2 usando ratio R/IR estándar"""
        try:
            ir_array = np.array(ir_values, dtype=float)
            red_array = np.array(red_values, dtype=float)
            
            # Validar entrada
            if len(ir_array) < 10 or len(red_array) < 10:
                return 98.0
            
            # Calcular componentes AC y DC
            ir_ac = np.std(ir_array)
            ir_dc = np.mean(ir_array)
            red_ac = np.std(red_array)
            red_dc = np.mean(red_array)
            
# Evitar división por cero
            if ir_dc <= 0 or red_dc <= 0 or ir_ac <= 0:
                return 98.0
            
            # Calcular ratio R
            r = (red_ac / red_dc) / (ir_ac / ir_dc)
            
            # Ecuación de calibración
            spo2 = 104 - 17 * r
            
            # Validar rango
            if 85 <= spo2 <= 100:
                return float(spo2)
            else:
                return 98.0
                
        except Exception as e:
            logger.warning(f"Error calculando SpO2: {e}")
            return 98.0
    
    def clear_patient_buffer(self, patient_id):
        """Limpiar completamente el buffer de un paciente"""
        if patient_id in self.patient_buffers:
            del self.patient_buffers[patient_id]
            logger.info(f"Buffer limpiado para paciente {patient_id}")

# Crear instancia global del buffer de muestras
sample_buffer = SampleBuffer()

def classify_pressure_level(sys_pressure, dia_pressure):
    """Clasificar nivel de presión arterial según guías médicas AHA/ESC"""
    try:
        sys_val, dia_val = float(sys_pressure), float(dia_pressure)
    except (ValueError, TypeError):
        return "Error"
    
    # Crisis de Hipertensión
    if sys_val > 180 or dia_val > 120:
        return "HT Crisis"
    
    # Hipertensión Etapa 2
    if sys_val >= 140 or dia_val >= 90:
        return "HT2"
    
    # Hipertensión Etapa 1
    if (130 <= sys_val <= 139) or (80 <= dia_val <= 89):
        return "HT1"
    
    # Elevada
    if 120 <= sys_val <= 129 and dia_val < 80:
        return "Elevada"
    
    # Normal
    if sys_val < 120 and dia_val < 80:
        return "Normal"
    
    return "Revisar"

@api_nodo_bp.route('/raw_data', methods=['POST'])
def receive_raw_data():
    """
    Endpoint principal para recibir datos del ESP32
    CORREGIDO: 50 muestras con limpieza automática
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
                "nivel": "HT Crisis", "muestras_recolectadas": 50,
                "calidad": "complete"
            }
            if hasattr(api_nodo_bp, 'websocket_handler'):
                api_nodo_bp.websocket_handler.emit_update(data)
            return jsonify(response)

        # Añadir muestra al buffer del paciente
        sample_count = sample_buffer.add_sample(patient_id, ir_value, red_value, timestamp)
        
        logger.info(f"Muestra añadida: {sample_count}/50 para paciente {patient_id}")

        # Respuesta por defecto mientras se recolectan muestras
        response = {
            "sys": 0, 
            "dia": 0, 
            "hr": 0, 
            "spo2": 0,
            "nivel": f"Recolectando datos... {sample_count}/50",
            "muestras_recolectadas": sample_count,
            "muestras_necesarias": 50,
            "calidad": "collecting"
        }

        # SOLO calcular cuando se tengan EXACTAMENTE 50 muestras
        if sample_count == 50:
            logger.info(f"50 muestras alcanzadas para paciente {patient_id} - Iniciando cálculos ML")
            
            # Obtener datos procesados para ML
            ml_data = sample_buffer.get_samples_for_ml(patient_id)
            
            if ml_data and hasattr(api_nodo_bp, 'ml_processor') and api_nodo_bp.ml_processor.is_ready():
                try:
                    # Realizar predicción ML con validación de datos
                    hr_calc = ml_data.get('hr_calculated', 75)
                    spo2_calc = ml_data.get('spo2_calculated', 98)
                    ir_mean = ml_data.get('ir_mean', 0)
                    red_mean = ml_data.get('red_mean', 0)
                    ir_std = ml_data.get('ir_std', 0)
                    red_std = ml_data.get('red_std', 0)
                    
                    # Validar datos antes de enviar a ML
                    if ir_mean > 0 and red_mean > 0:
                        # Usar el método corregido que devuelve 3 valores
                        result = api_nodo_bp.ml_processor.predict_pressure(
                            hr_calc, spo2_calc, ir_mean, red_mean, ir_std, red_std
                        )
                        
                        # Manejar respuesta de ML (puede ser 2 o 3 valores)
                        if isinstance(result, tuple) and len(result) >= 2:
                            sys_pred = result[0]
                            dia_pred = result[1]
                            hr_final = result[2] if len(result) > 2 else hr_calc
                        else:
                            logger.error("Respuesta ML inválida")
                            sys_pred, dia_pred, hr_final = 0, 0, hr_calc
                        
# En modules/api_nodo_datos.py
# Cambiar la sección donde guarda en BD (línea aproximada 280-295)

                        # Validar predicciones
                        if sys_pred > 0 and dia_pred > 0:
                            # Actualizar respuesta con resultados ML
                            response.update({
                                "sys": round(float(sys_pred), 1),
                                "dia": round(float(dia_pred), 1),
                                "hr": round(float(hr_final), 0),
                                "spo2": round(float(spo2_calc), 0),
                                "nivel": classify_pressure_level(sys_pred, dia_pred),
                                "calidad": "complete"
                            })
                            
                            logger.info(f"Predicción ML exitosa - SYS:{sys_pred} DIA:{dia_pred}")
                            
                            # GUARDAR EN BASE DE DATOS - FORZAR SIEMPRE
                            try:
                                if hasattr(api_nodo_bp, 'db_manager'):
                                    measurement_data = {
                                        'id_paciente': patient_id,
                                        'sys': float(sys_pred),                     
                                        'dia': float(dia_pred),                     
                                        'hr_ml': float(hr_final),
                                        'spo2_ml': float(spo2_calc),
                                        'nivel': str(response["nivel"])           
                                    }
                                    
                                    # Usar save directamente en lugar de async
                                    api_nodo_bp.db_manager._save_measurement_sync(measurement_data)
                                    logger.info(f"Medición FORZADA guardada en BD: SYS={sys_pred}, DIA={dia_pred}")
                                else:
                                    logger.error("db_manager no disponible")
                            except Exception as e:
                                logger.error(f"Error FORZANDO guardado BD: {e}")
                            
                            # Verificar alertas (resto del código igual)
                            if hasattr(api_nodo_bp, 'alert_system'):
                                alert_data = {
                                    'patient_id': patient_id,
                                    'nivel': response["nivel"],
                                    'sys': sys_pred,
                                    'dia': dia_pred,
                                    'hr': hr_final,
                                    'spo2': spo2_calc
                                }
                                api_nodo_bp.alert_system.check_and_send_alert(alert_data)
                            
                            # Notificar vía WebSocket
                            if hasattr(api_nodo_bp, 'websocket_handler'):
                                api_nodo_bp.websocket_handler.emit_new_record_saved()
                        else:
                            logger.error("Predicciones ML inválidas")
                            response["nivel"] = "Error predicción"
                            response["calidad"] = "error"
                    else:
                        logger.error("Datos de entrada ML inválidos")
                        response["nivel"] = "Datos inválidos"
                        response["calidad"] = "error"
                    
                    # NUEVO: LIMPIEZA AUTOMÁTICA DEL BUFFER DESPUÉS DE PREDICCIÓN
                    sample_buffer.clear_patient_buffer(patient_id)
                    logger.info(f"Buffer limpiado automáticamente para paciente {patient_id}")
                    
                except Exception as e:
                    logger.error(f"Error en predicción ML: {e}")
                    response["nivel"] = "Error ML"
                    response["calidad"] = "error"
                    
                    # Limpiar buffer incluso en caso de error
                    sample_buffer.clear_patient_buffer(patient_id)
                    logger.info(f"Buffer limpiado por error para paciente {patient_id}")
            else:
                logger.warning("ML processor no disponible")
                response["nivel"] = "ML no disponible"
                response["calidad"] = "error"
        
        # Notificar actualización vía WebSocket
        if hasattr(api_nodo_bp, 'websocket_handler'):
            update_data = {**data, **response}
            api_nodo_bp.websocket_handler.emit_update(update_data)

        logger.info(f"Respuesta enviada - Muestras:{sample_count}/50 Estado:{response['calidad']}")
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
            "samples_needed": 50,
            "ready_for_ml": has_enough,
            "progress_percentage": (sample_count / 50) * 100,
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
        all_patients = {}
        for patient_id, buffer in sample_buffer.patient_buffers.items():
            all_patients[patient_id] = {
                "sample_count": len(buffer),
                "progress_percentage": (len(buffer) / 50) * 100,
                "ready_for_ml": len(buffer) >= 50
            }
        
        return jsonify({
            "active_patients": len(all_patients),
            "patient_buffers": all_patients,
            "target_samples": 50,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error obteniendo estado de buffers: {e}")
        return jsonify({"error": "Error interno"}), 500

@api_nodo_bp.route('/clear_buffer/<int:patient_id>', methods=['POST'])
def clear_patient_buffer_endpoint(patient_id):
    """Endpoint para limpiar manualmente el buffer de un paciente"""
    try:
        sample_buffer.clear_patient_buffer(patient_id)
        logger.info(f"Buffer limpiado manualmente para paciente {patient_id}")
        return jsonify({
            "success": True,
            "message": f"Buffer del paciente {patient_id} limpiado",
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        logger.error(f"Error limpiando buffer paciente {patient_id}: {e}")
        return jsonify({"error": "Error interno"}), 500

def init_api_module(app):
    """Inicializar el módulo API con la aplicación Flask"""
    logger.info("Inicializando módulo API de nodos con sistema de 50 muestras y limpieza automática")
    logger.info("Módulo API de nodos inicializado correctamente")

# Inicializar automáticamente cuando se importe
init_api_module(None)
