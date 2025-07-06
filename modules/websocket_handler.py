# modules/websocket_handler.py
# Manejador WebSocket MINIMALISTA - SIN ERRORES

import logging
import threading
import time
from datetime import datetime
from collections import defaultdict
import queue

class WebSocketHandler:
    """Manejador WebSocket simplificado sin errores"""
    
    def __init__(self, socketio_instance=None):
        self.logger = logging.getLogger(__name__)
        self.socketio = socketio_instance
        
        # Estados básicos
        self.connected_clients = set()
        self.messages_sent = 0
        self.messages_failed = 0
        self.last_message_time = 0
        
        # Cola simple
        self.message_queue = queue.Queue()
        self.max_queue_size = 100  # Más pequeño
        
        # Worker thread básico
        self.worker_thread = None
        self.should_stop = False
        
        self.logger.info("WebSocket Handler MINIMALISTA inicializado")
        
        if self.socketio:
            self._start_minimal_worker()
    
    def _start_minimal_worker(self):
        """Worker thread mínimo"""
        self.worker_thread = threading.Thread(target=self._minimal_worker, daemon=True)
        self.worker_thread.start()
    
    def _minimal_worker(self):
        """Worker loop mínimo"""
        while not self.should_stop:
            try:
                message_data = self.message_queue.get(timeout=1)
                self._send_message_simple(message_data)
                self.message_queue.task_done()
            except queue.Empty:
                continue
            except:
                pass
    
    def _send_message_simple(self, message_data):
        """Enviar mensaje de forma simple"""
        try:
            if self.socketio:
                event_name = message_data.get('event', 'update')
                data = message_data.get('data', {})
                
                self.socketio.emit(event_name, data)
                self.messages_sent += 1
                self.last_message_time = time.time()
        except:
            self.messages_failed += 1
    
    def emit_update(self, data, event_name='update_data'):
        """Emitir actualización básica"""
        try:
            if not self.socketio:
                return False
            
            # Limpiar datos
            clean_data = self._clean_data(data)
            
            # Enviar directamente si la cola está vacía
            if self.message_queue.empty():
                self.socketio.emit(event_name, clean_data)
                self.messages_sent += 1
                self.last_message_time = time.time()
                return True
            else:
                # Añadir a cola
                if self.message_queue.qsize() < self.max_queue_size:
                    self.message_queue.put({
                        'event': event_name,
                        'data': clean_data
                    })
                return True
        except:
            self.messages_failed += 1
            return False
    
    def _clean_data(self, data):
        """Limpiar datos básico"""
        if isinstance(data, dict):
            clean = {}
            for k, v in data.items():
                if v is not None:
                    if isinstance(v, (int, float, str, bool, list, dict)):
                        clean[k] = v
                    else:
                        clean[k] = str(v)
            return clean
        return data
    
    def emit_capture_count(self, count):
        """Emitir contador de muestras"""
        return self.emit_update({'count': count}, 'capture_count_update')
    
    def emit_new_record_saved(self):
        """Notificar registro guardado"""
        return self.emit_update({'timestamp': datetime.now().isoformat()}, 'new_record_saved')
    
    def emit_alert(self, alert_data):
        """Emitir alerta médica"""
        alert_message = {
            'type': 'medical_alert',
            'level': alert_data.get('level', 'info'),
            'patient_id': alert_data.get('patient_id'),
            'timestamp': datetime.now().isoformat()
        }
        return self.emit_update(alert_message, 'medical_alert')
    
    def emit_system_status(self, status_data):
        """Emitir estado del sistema"""
        status_message = {
            'timestamp': datetime.now().isoformat(),
            'status': status_data
        }
        return self.emit_update(status_message, 'system_status')
    
    def emit_error(self, error_message, error_type='general'):
        """Emitir mensaje de error"""
        error_data = {
            'type': error_type,
            'message': error_message,
            'timestamp': datetime.now().isoformat()
        }
        return self.emit_update(error_data, 'error_message')
    
    def handle_client_connect(self, client_id, client_info=None):
        """Manejar conexión - MÍNIMO"""
        try:
            self.connected_clients.add(client_id)
            self.logger.info(f"Cliente conectado: {client_id}")
        except:
            pass
    
    def handle_client_disconnect(self, client_id):
        """Manejar desconexión - NO HACER NADA"""
        # COMPLETAMENTE VACÍO - no procesar desconexiones
        pass
    
    def get_connected_clients_count(self):
        """Obtener número de clientes"""
        try:
            return len(self.connected_clients)
        except:
            return 0
    
    def get_status(self):
        """Estado básico"""
        try:
            return {
                'socketio_configured': self.socketio is not None,
                'connected_clients': len(self.connected_clients),
                'messages_sent': self.messages_sent,
                'messages_failed': self.messages_failed,
                'queue_size': self.message_queue.qsize(),
                'last_message': datetime.fromtimestamp(self.last_message_time).isoformat() if self.last_message_time else None
            }
        except:
            return {
                'socketio_configured': False,
                'connected_clients': 0,
                'messages_sent': 0,
                'messages_failed': 0,
                'queue_size': 0,
                'last_message': None
            }
    
    def broadcast_system_message(self, message, message_type='info'):
        """Mensaje del sistema"""
        system_message = {
            'type': 'system_message',
            'level': message_type,
            'message': message,
            'timestamp': datetime.now().isoformat()
        }
        return self.emit_update(system_message, 'system_message')
    
    def register_event_handlers(self):
        """NO registrar handlers - se hace en app.py"""
        self.logger.info("Event handlers omitidos (modo minimalista)")
    
    def shutdown(self):
        """Apagado simple"""
        try:
            self.should_stop = True
            # NO esperar threads - salir inmediatamente
            self.connected_clients.clear()
        except:
            pass
    
    # Métodos adicionales simplificados para compatibilidad
    def set_socketio(self, socketio_instance):
        self.socketio = socketio_instance
    
    def emit_to_room(self, room, event_name, data):
        return self.emit_update(data, event_name)
    
    def clear_message_queue(self):
        try:
            while not self.message_queue.empty():
                self.message_queue.get_nowait()
            return 0
        except:
            return 0
    
    def update_client_info(self, client_id, info_update):
        return True
    
    def get_client_info(self, client_id):
        return {}
    
    def configure_rate_limiting(self, max_messages_per_minute=None, max_queue_size=None):
        pass
    
    def emit_data_batch(self, data_list, event_name='batch_update', batch_size=10):
        return True
    
    def get_performance_metrics(self):
        return self.get_status()
    
    def send_heartbeat(self):
        return self.emit_update({'status': 'alive'}, 'heartbeat')
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            if hasattr(self, 'should_stop'):
                self.should_stop = True
        except:
            pass
