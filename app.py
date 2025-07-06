# app.py - VERSION MODULAR COMPLETA SIN ERRORES WEBSOCKET - CORREGIDA
# Aplicación principal usando arquitectura modular

import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, redirect, url_for, jsonify
from flask_socketio import SocketIO
import os
import time
import logging
from datetime import datetime
import threading
import sys
import warnings

# Suprimir warnings y errores de eventlet/socketio
warnings.filterwarnings("ignore")
logging.getLogger('socketio').setLevel(logging.WARNING)
logging.getLogger('engineio').setLevel(logging.WARNING) 
logging.getLogger('eventlet').setLevel(logging.WARNING)

# Importar módulos especializados
from modules.ml_processor import MLProcessor
from modules.database_manager import DatabaseManager
from modules.alert_system import AlertSystem
from modules.data_collector import RealisticTrainingCollector  # CORREGIDO
from modules.websocket_handler import WebSocketHandler
from modules.api_nodo_datos import api_nodo_bp

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='[%(name)s] %(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

class MedicalMonitorApp:
    """Aplicación principal del sistema de monitoreo médico"""
    
    def __init__(self):
        self.app = Flask(__name__)
        self.app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev_secret_key_change_in_production')
        
        # Inicializar SocketIO con configuración anti-errores
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            async_mode='eventlet',
            logger=False,
            engineio_logger=False,
            ping_timeout=60,
            ping_interval=25,
            max_http_buffer_size=100000,
            always_connect=False,
            transports=['websocket', 'polling'],
            allow_upgrades=True,
            cookie=None
        )
        
        # Inicializar módulos especializados
        self.ml_processor = MLProcessor()
        self.db_manager = DatabaseManager()
        self.alert_system = AlertSystem()
        self.data_collector = RealisticTrainingCollector()  # CORREGIDO
        self.websocket_handler = WebSocketHandler(self.socketio)
        
        # Variables de estado
        self.system_start_time = time.time()
        self.last_db_save_time = 0
        self.save_interval = 5  # segundos
        
        # Buffer para datos del ESP32
        self.buffer_datos_entrenamiento = []
        self.capture_lock_file = "capture.lock"
        
        # Configurar conexiones entre módulos
        self._setup_module_connections()
        
        # Registrar rutas y eventos
        self._register_routes()
        self._register_socketio_events()
        
        logger.info("Sistema de monitoreo médico inicializado")
    
    def _setup_module_connections(self):
        """Configurar conexiones entre módulos"""
        # Conectar API con otros módulos
        api_nodo_bp.ml_processor = self.ml_processor
        api_nodo_bp.db_manager = self.db_manager
        api_nodo_bp.alert_system = self.alert_system
        api_nodo_bp.websocket_handler = self.websocket_handler
        
        # Registrar blueprint de API
        self.app.register_blueprint(api_nodo_bp, url_prefix='/api')
        
        # Configurar WebSocket handler
        self.websocket_handler.register_event_handlers()
        
        logger.info("Conexiones entre módulos configuradas")
    
    def _register_routes(self):
        """Registrar rutas de la aplicación"""
        
        @self.app.route("/")
        def home():
            """Página principal del panel de control"""
            return render_template("index.html")
        
        @self.app.route("/api/data", methods=["POST"])
        def recibir_datos():
            """Endpoint legacy para compatibilidad con ESP32 anterior"""
            return self._handle_legacy_esp32_data()
        
        @self.app.route("/api/start_capture", methods=["POST"])
        def start_capture():
            """Iniciar captura de datos para entrenamiento - CORREGIDO"""
            try:
                # Usar el método correcto del DataCollector
                result = self.data_collector.start_training_session()
                
                logger.info("Sesión de entrenamiento iniciada")
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error iniciando captura: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/stop_capture", methods=["POST"])
        def stop_capture():
            """Detener captura de datos - CORREGIDO"""
            try:
                # Usar el método correcto del DataCollector  
                result = self.data_collector.stop_training_session()
                
                logger.info(f"Captura detenida")
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error deteniendo captura: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/save_training_data", methods=["POST"])
        def save_training_data():
            """Guardar datos de entrenamiento - CORREGIDO"""
            try:
                ref_data = request.get_json()
                if not ref_data:
                    return jsonify({"error": "Datos de referencia requeridos"}), 400
                
                # Usar el método correcto del DataCollector
                result = self.data_collector.save_training_data(ref_data)
                
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error guardando datos entrenamiento: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/training/add_sample", methods=["POST"])  
        def add_training_sample():
            """Añadir muestra durante entrenamiento"""
            try:
                data = request.get_json()
                if not data:
                    return jsonify({"error": "No data"}), 400
                    
                # Añadir muestra al entrenamiento
                result = self.data_collector.add_sample(data)
                
                # Notificar conteo vía WebSocket
                if result.get('success'):
                    self.websocket_handler.emit_capture_count(result.get('total_samples', 0))
                
                return jsonify(result)
                
            except Exception as e:
                logger.error(f"Error añadiendo muestra: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/ultimas_mediciones")
        def get_ultimas_mediciones():
            """Obtener últimas mediciones de la base de datos"""
            try:
                records = self.db_manager.get_latest_measurements(limit=20)
                return jsonify(records)
            except Exception as e:
                logger.error(f"Error obteniendo mediciones: {e}")
                return jsonify([])
        
        @self.app.route("/api/mediciones_recientes")
        def get_mediciones_recientes():
            """Obtener últimas mediciones para el panel"""
            try:
                limit = request.args.get('limit', 20, type=int)
                records = self.db_manager.get_latest_measurements(limit=limit)
                
                # Formatear para el panel usando los nombres exactos de tu tabla
                mediciones = []
                for record in records:
                    mediciones.append({
                        'id': record.get('id'),
                        'patient_id': record.get('id_paciente'),  # id_paciente
                        'sys': float(record.get('sys', 0)),       # sys
                        'dia': float(record.get('dia', 0)),       # dia
                        'hr': float(record.get('hr_ml', 0)),      # hr_ml
                        'spo2': float(record.get('spo2_ml', 0)),  # spo2_ml
                        'nivel': record.get('nivel', '---'),      # nivel
                        'timestamp': str(record.get('timestamp_medicion', '')),
                        'fecha_formateada': str(record.get('timestamp_medicion', ''))[:19]
                    })
                
                return jsonify({
                    'success': True,
                    'mediciones': mediciones,
                    'count': len(mediciones)
                })
                
            except Exception as e:
                logger.error(f"Error obteniendo mediciones recientes: {e}")
                return jsonify({
                    'success': False, 
                    'error': str(e),
                    'mediciones': []
                })
        
        @self.app.route("/api/test_alert", methods=['POST'])
        def test_alert():
            """Endpoint de prueba para alertas"""
            try:
                data = request.get_json()
                if not data or "sys" not in data or "dia" not in data:
                    return jsonify({"error": "Datos incompletos"}), 400
                
                # Procesar alerta de prueba
                test_data = {
                    "patient_id": data.get("id_paciente", 99),
                    "sys": float(data["sys"]),
                    "dia": float(data["dia"]),
                    "hr": data.get("hr", 0),
                    "spo2": data.get("spo2", 0),
                    "nivel": self._classify_pressure_level(float(data["sys"]), float(data["dia"]))
                }
                
                # Guardar en BD
                self.db_manager.save_measurement_async(test_data)
                
                # Enviar alerta
                self.alert_system.check_and_send_alert(test_data)
                
                # Notificar vía WebSocket
                self.websocket_handler.emit_new_record_saved()
                
                return jsonify({
                    "status": "success",
                    "message": "Alerta de prueba procesada",
                    "data": test_data
                })
            
            except Exception as e:
                logger.error(f"Error en test de alerta: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/system_status")
        def get_system_status():
            """Obtener estado completo del sistema"""
            try:
                status = {
                    "timestamp": datetime.now().isoformat(),
                    "uptime_hours": (time.time() - self.system_start_time) / 3600,
                    "modules": {
                        "ml_processor": self.ml_processor.get_status(),
                        "database": self.db_manager.get_system_health(),
                        "alerts": self.alert_system.get_status(),
                        "websocket": self.websocket_handler.get_status(),
                        "data_collector": self.data_collector.get_training_status()
                    }
                }
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado del sistema: {e}")
                return jsonify({"error": str(e)}), 500
        
        logger.info("Rutas principales registradas")
    
    def _register_socketio_events(self):
        """Registrar eventos de SocketIO sin errores"""
        
        @self.socketio.on('connect')
        def handle_connect():
            # Mínimo procesamiento
            pass
        
        @self.socketio.on('disconnect')
        def handle_disconnect():
            # COMPLETAMENTE VACÍO - no procesar desconexiones
            pass
        
        @self.socketio.on('request_system_status')
        def handle_status_request():
            try:
                status = {
                    "ml_ready": self.ml_processor.is_ready(),
                    "db_connected": self.db_manager.is_connected(),
                    "alerts_configured": self.alert_system.is_configured(),
                    "capture_active": self.data_collector.get_training_status().get('active', False),
                    "connected_clients": self.websocket_handler.get_connected_clients_count()
                }
                self.socketio.emit('system_status', status)
            except:
                pass
        
        logger.info("Eventos SocketIO registrados")
    
    def _handle_legacy_esp32_data(self):
        """Manejar datos del ESP32 en formato legacy"""
        try:
            data = request.get_json()
            if not data:
                return jsonify({"error": "No JSON data"}), 400
            
            # CORREGIDO: Verificar si estamos en modo entrenamiento
            training_status = self.data_collector.get_training_status()
            if training_status.get('active', False):
                # MODO ENTRENAMIENTO: Añadir muestra al data_collector
                try:
                    result = self.data_collector.add_sample(data)
                    
                    # Notificar conteo vía WebSocket
                    if result.get('success'):
                        self.websocket_handler.emit_capture_count(result.get('total_samples', 0))
                    
                    return jsonify({
                        "status": "entrenamiento",
                        "message": f"Muestra añadida. Total: {result.get('total_samples', 0)}",
                        "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
                        "nivel": "Entrenamiento activo"
                    })
                    
                except Exception as e:
                    logger.error(f"Error en modo entrenamiento: {e}")
                    return jsonify({
                        "status": "error_entrenamiento",
                        "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
                        "nivel": "Error entrenamiento"
                    })
            
            # MODO PREDICCIÓN NORMAL (redirigir al endpoint modular)
            return self._process_esp32_data_legacy(data)
        
        except Exception as e:
            logger.error(f"Error procesando datos legacy: {e}")
            return jsonify({
                "sys": 0, "dia": 0, "hr": 0, "spo2": 0,
                "nivel": "Error servidor"
            }), 500
    
    def _process_esp32_data_legacy(self, data):
        """Procesar datos ESP32 en modo legacy"""
        patient_id = data.get('id_paciente', 1)
        
        # Modo prueba buzzer
        if patient_id == 999:
            response = {"sys": 185, "dia": 125, "hr": 99, "spo2": 99, "nivel": "HT Crisis"}
            self.websocket_handler.emit_update(data)
            return jsonify(response)
        
        # Modo predicción normal
        response = {"sys": 0, "dia": 0, "hr": 0, "spo2": 0, "nivel": "Sin datos"}
        
        # Verificar señal válida
        ir_value = float(data.get("ir", 0))
        if ir_value > 50000:
            if self.ml_processor.is_ready():
                try:
                    # Preparar features para ML
                    hr = float(data.get("hr_promedio", 0))
                    spo2 = float(data.get("spo2_sensor", 0))
                    ir_mean = float(data.get("ir", 0))
                    red_mean = float(data.get("red", 0))
                    
                    # Calcular std (simplificado para legacy)
                    ir_std = ir_mean * 0.02  # Aproximación
                    red_std = red_mean * 0.02
                    
                    # Predicción ML
                    sys_pred, dia_pred = self.ml_processor.predict_pressure(
                        hr, spo2, ir_mean, red_mean, ir_std, red_std
                    )
                    
                    response.update({
                        "sys": round(sys_pred, 2),
                        "dia": round(dia_pred, 2),
                        "hr": round(hr, 2),
                        "spo2": round(spo2, 2),
                        "nivel": self._classify_pressure_level(sys_pred, dia_pred)
                    })
                    
                    # Guardar en BD periódicamente
                    if (time.time() - self.last_db_save_time) >= self.save_interval:
                        measurement_data = {
                            'id_paciente': patient_id,
                            'sys': sys_pred,
                            'dia': dia_pred,
                            'hr_ml': hr,
                            'spo2_ml': spo2,
                            'nivel': response["nivel"]
                        }
                        self.db_manager.save_measurement_async(measurement_data)
                        self.websocket_handler.emit_new_record_saved()
                        self.last_db_save_time = time.time()
                    
                    # Verificar alertas
                    alert_data = {
                        'patient_id': patient_id,
                        'nivel': response["nivel"],
                        'sys': sys_pred,
                        'dia': dia_pred,
                        'hr': hr,
                        'spo2': spo2
                    }
                    self.alert_system.check_and_send_alert(alert_data)
                
                except Exception as e:
                    logger.error(f"Error en predicción ML: {e}")
                    response["nivel"] = "Error ML"
        
        # Enviar datos vía WebSocket
        self.websocket_handler.emit_update({**data, **response})
        
        return jsonify(response)
    
    def _classify_pressure_level(self, sys_pressure, dia_pressure):
        """Clasificar nivel de presión arterial según guías médicas AHA/ESC"""
        if sys_pressure is None or dia_pressure is None:
            return "N/A"
        
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
    
    def run(self, host='0.0.0.0', port=None, debug=False):
        """Ejecutar la aplicación"""
        if port is None:
            port = int(os.environ.get("PORT", 10000))
        
        logger.info(f"Iniciando servidor en {host}:{port}")
        logger.info(f"Estado módulos - ML: {self.ml_processor.is_ready()}, "
                   f"BD: {self.db_manager.is_connected()}, "
                   f"Alertas: {self.alert_system.is_configured()}")
        
        # Crear tablas de BD si no existen
        if self.db_manager.is_connected():
            self.db_manager.create_tables_if_not_exist()
        
        # Ejecutar servidor con configuración anti-errores
        try:
            import socket
            socket.setdefaulttimeout(30)
            
            self.socketio.run(
                self.app,
                host=host,
                port=port,
                debug=False,
                use_reloader=False,
                log_output=False
            )
        except Exception as e:
            logger.error(f"Error ejecutando servidor: {e}")
            self.shutdown()
    
    def shutdown(self):
        """Apagar sistema de forma segura"""
        try:
            logger.info("Iniciando apagado del sistema...")
            
            # Apagar módulos en orden
            self.websocket_handler.shutdown()
            self.alert_system.shutdown()
            self.db_manager.close_connections()
            
            # Limpiar archivos temporales
            if os.path.exists(self.capture_lock_file):
                os.remove(self.capture_lock_file)
            
            logger.info("Sistema apagado correctamente")
        except Exception:
            pass

# Crear instancia global de la aplicación
medical_app = MedicalMonitorApp()
app = medical_app.app
socketio = medical_app.socketio

# Función de compatibilidad para gunicorn
def create_app():
    """Factory function para compatibilidad con algunos servidores"""
    return medical_app.app

# Clase para ocultar errores de WebSocket
class QuietStderr:
    def write(self, s):
        if "Bad file descriptor" not in s and "socket shutdown error" not in s:
            sys.__stderr__.write(s)
    
    def flush(self):
        sys.__stderr__.flush()

if __name__ == "__main__":
    # CONFIGURACIÓN FINAL ANTI-ERRORES
    warnings.filterwarnings("ignore")
    
    # Configurar stderr para no mostrar errores de eventlet
    sys.stderr = QuietStderr()
    
    try:
        medical_app.run(debug=False)
    except KeyboardInterrupt:
        medical_app.shutdown()
    except Exception as e:
        logger.error(f"Error crítico: {e}")
        medical_app.shutdown()
