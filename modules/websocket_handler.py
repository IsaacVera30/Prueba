# app.py - VERSION MODULAR
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

# Importar módulos especializados
from modules.ml_processor import MLProcessor
from modules.database_manager import DatabaseManager
from modules.alert_system import AlertSystem
from modules.data_collector import DataCollector
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
        
        # Inicializar SocketIO con configuración mejorada
        self.socketio = SocketIO(
            self.app, 
            cors_allowed_origins="*", 
            async_mode='eventlet',
            logger=False,
            engineio_logger=False,
            ping_timeout=30,        # Timeout aumentado
            ping_interval=10,       # Intervalo de ping
            max_http_buffer_size=1000000  # Buffer más grande
        )
        
        # Inicializar módulos especializados
        self.ml_processor = MLProcessor()
        self.db_manager = DatabaseManager()
        self.alert_system = AlertSystem()
        self.data_collector = DataCollector()
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
            """Iniciar captura de datos para entrenamiento"""
            try:
                result = self.data_collector.start_capture()
                
                # Crear archivo lock para compatibilidad
                with open(self.capture_lock_file, "w") as f:
                    f.write("capturing")
                
                self.buffer_datos_entrenamiento = []
                
                logger.info("Captura de entrenamiento iniciada")
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error iniciando captura: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/stop_capture", methods=["POST"])
        def stop_capture():
            """Detener captura de datos"""
            try:
                result = self.data_collector.stop_capture()
                
                # Añadir información del buffer local
                result["muestras_en_buffer"] = len(self.buffer_datos_entrenamiento)
                
                logger.info(f"Captura detenida. {len(self.buffer_datos_entrenamiento)} muestras")
                return jsonify(result)
            
            except Exception as e:
                logger.error(f"Error deteniendo captura: {e}")
                return jsonify({"error": str(e)}), 500
        
        @self.app.route("/api/save_training_data", methods=["POST"])
        def save_training_data():
            """Guardar datos de entrenamiento"""
            try:
                if not os.path.exists(self.capture_lock_file):
                    return jsonify({"error": "La captura no está activa"}), 400
                
                ref_data = request.get_json()
                if not ref_data:
                    return jsonify({"error": "Datos de referencia requeridos"}), 400
                
                # Procesar buffer y guardar
                self._process_and_save_training_data(ref_data)
                
                return jsonify({
                    "status": "success",
                    "message": "Datos de entrenamiento guardados",
                    "samples_processed": len(self.buffer_datos_entrenamiento)
                })
            
            except Exception as e:
                logger.error(f"Error guardando datos entrenamiento: {e}")
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
                        "data_collector": self.data_collector.get_status()
                    }
                }
                return jsonify(status)
            except Exception as e:
                logger.error(f"Error obteniendo estado del sistema: {e}")
                return jsonify({"error": str(e)}), 500
        
        logger.info("Rutas principales registradas")
    
    def _register_socketio_events(self):
        """Registrar eventos de SocketIO"""
        
        @self.socketio.on('connect')
        def handle_connect():
            try:
                client_id = request.sid
                self.websocket_handler.handle_client_connect(client_id)
                logger.info(f"Cliente WebSocket conectado: {client_id}")
            except Exception as e:
                logger.debug(f"
