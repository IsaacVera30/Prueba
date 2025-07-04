# modules/alert_system.py
# Sistema especializado para manejo de alertas médicas

import os
import threading
import time
import logging
import urllib.request
import urllib.parse
from datetime import datetime, timedelta
from collections import defaultdict
import queue

class AlertSystem:
    """Sistema de alertas médicas con diferentes canales de notificación"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración de WhatsApp (CallMeBot)
        self.whatsapp_api_key = os.environ.get("CALLMEBOT_API_KEY")
        self.whatsapp_phone = os.environ.get("CALLMEBOT_PHONE_NUMBER")
        
        # Configuración de email (si está disponible)
        self.email_enabled = False  # Implementar si es necesario
        
        # Control de rate limiting (evitar spam)
        self.alert_history = defaultdict(list)
        self.min_alert_interval = 300  # 5 minutos entre alertas del mismo tipo
        self.max_alerts_per_hour = 10
        
        # Cola de alertas para procesamiento asíncrono
        self.alert_queue = queue.Queue()
        self.worker_thread = None
        self.should_stop = False
        
        # Métricas
        self.alerts_sent = 0
        self.alerts_blocked = 0
        self.last_alert_time = 0
        
        # Lock para thread safety
        self.alert_lock = threading.Lock()
        
        # Configuración de niveles de alerta
        self.alert_levels = {
            "HT Crisis": {
                "priority": "critical",
                "channels": ["whatsapp", "log"],
                "template": "CRISIS HIPERTENSIVA\nPaciente: {patient_id}\nSYS: {sys} mmHg\nDIA: {dia} mmHg\nTiempo: {timestamp}\nATENCION MEDICA INMEDIATA!"
            },
            "HT2": {
                "priority": "high", 
                "channels": ["whatsapp", "log"],
                "template": "Hipertension Grado 2\nPaciente: {patient_id}\nSYS: {sys} mmHg\nDIA: {dia} mmHg\nTiempo: {timestamp}\nConsultar medico pronto."
            },
            "HT1": {
                "priority": "medium",
                "channels": ["log"],
                "template": "Hipertension Grado 1\nPaciente: {patient_id}\nSYS: {sys} mmHg\nDIA: {dia} mmHg\nTiempo: {timestamp}\nMonitorear de cerca."
            }
        }
        
        # Inicializar sistema
        self._start_worker_thread()
        self._log_configuration()
    
    def _log_configuration(self):
        """Mostrar configuración del sistema de alertas"""
        self.logger.info("Sistema de alertas inicializado")
        
        if self.whatsapp_api_key and self.whatsapp_phone:
            self.logger.info("WhatsApp configurado")
        else:
            self.logger.warning("WhatsApp no configurado (falta API key o teléfono)")
        
        self.logger.info(f"Niveles de alerta configurados: {list(self.alert_levels.keys())}")
    
    def _start_worker_thread(self):
        """Iniciar hilo trabajador para envío de alertas"""
        self.worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self.worker_thread.start()
        self.logger.info("Hilo trabajador de alertas iniciado")
    
    def _worker_loop(self):
        """Loop principal del hilo trabajador"""
        while not self.should_stop:
            try:
                # Obtener alerta de la cola
                alert_data = self.alert_queue.get(timeout=1)
                
                # Procesar alerta
                self._process_alert(alert_data)
                
                # Marcar tarea como completada
                self.alert_queue.task_done()
                
            except queue.Empty:
                continue
            except Exception as e:
                self.logger.error(f"Error en worker de alertas: {e}")
    
    def check_and_send_alert(self, measurement_data):
        """
        Verificar medición y enviar alerta si es necesario
        
        Args:
            measurement_data: Dict con datos de la medición
        """
        nivel = measurement_data.get('nivel', '')
        patient_id = measurement_data.get('patient_id', 'Desconocido')
        
        # Solo alertar para niveles críticos
        if nivel not in self.alert_levels:
            return
        
        # Verificar rate limiting
        if not self._should_send_alert(nivel, patient_id):
            self.alerts_blocked += 1
            self.logger.debug(f"Alerta bloqueada por rate limiting: {nivel}")
            return
        
        # Preparar datos de alerta
        alert_data = {
            'level': nivel,
            'patient_id': patient_id,
            'sys': measurement_data.get('sys', 0),
            'dia': measurement_data.get('dia', 0),
            'hr': measurement_data.get('hr', 0),
            'spo2': measurement_data.get('spo2', 0),
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'priority': self.alert_levels[nivel]['priority']
        }
        
        # Añadir a cola para procesamiento asíncrono
        self.alert_queue.put(alert_data)
        self.logger.info(f"Alerta añadida a cola: {nivel} - Paciente {patient_id}")
    
    def _should_send_alert(self, nivel, patient_id):
        """Verificar si se debe enviar alerta basado en rate limiting"""
        with self.alert_lock:
            current_time = time.time()
            alert_key = f"{nivel}_{patient_id}"
            
            # Limpiar alertas antiguas
            self._cleanup_old_alerts(current_time)
            
            # Verificar interval mínimo
            if alert_key in self.alert_history:
                last_alert = max(self.alert_history[alert_key])
                if current_time - last_alert < self.min_alert_interval:
                    return False
            
            # Verificar máximo por hora
            recent_alerts = [
                t for alerts in self.alert_history.values() 
                for t in alerts 
                if current_time - t < 3600
            ]
            
            if len(recent_alerts) >= self.max_alerts_per_hour:
                return False
            
            # Registrar nueva alerta
            self.alert_history[alert_key].append(current_time)
            return True
    
    def _cleanup_old_alerts(self, current_time):
        """Limpiar alertas antiguas del historial"""
        cutoff_time = current_time - 3600  # 1 hora
        
        for key in list(self.alert_history.keys()):
            self.alert_history[key] = [
                t for t in self.alert_history[key] 
                if t > cutoff_time
            ]
            
            # Eliminar claves vacías
            if not self.alert_history[key]:
                del self.alert_history[key]
    
    def _process_alert(self, alert_data):
        """Procesar una alerta específica"""
        try:
            nivel = alert_data['level']
            config = self.alert_levels[nivel]
            channels = config['channels']
            template = config['template']
            
            # Formatear mensaje
            message = template.format(**alert_data)
            
            # Enviar por cada canal configurado
            success_channels = []
            failed_channels = []
            
            for channel in channels:
                try:
                    if channel == "whatsapp":
                        success = self._send_whatsapp_alert(message)
                    elif channel == "log":
                        success = self._send_log_alert(message, alert_data['priority'])
                    elif channel == "email":
                        success = self._send_email_alert(message, alert_data)
                    else:
                        success = False
                        self.logger.warning(f"Canal desconocido: {channel}")
                    
                    if success:
                        success_channels.append(channel)
                    else:
                        failed_channels.append(channel)
                        
                except Exception as e:
                    self.logger.error(f"Error enviando por {channel}: {e}")
                    failed_channels.append(channel)
            
            # Actualizar métricas
            if success_channels:
                self.alerts_sent += 1
                self.last_alert_time = time.time()
                self.logger.info(f"Alerta enviada: {nivel} - Canales: {success_channels}")
            
            if failed_channels:
                self.logger.warning(f"Canales fallidos: {failed_channels}")
                
        except Exception as e:
            self.logger.error(f"Error procesando alerta: {e}")
    
    def _send_whatsapp_alert(self, message):
        """Enviar alerta por WhatsApp usando CallMeBot"""
        if not (self.whatsapp_api_key and self.whatsapp_phone):
            self.logger.debug("WhatsApp no configurado")
            return False
        
        try:
            # Preparar URL
            encoded_message = urllib.parse.quote(message)
            url = (
                f"https://api.callmebot.com/whatsapp.php?"
                f"phone={self.whatsapp_phone}&"
                f"text={encoded_message}&"
                f"apikey={self.whatsapp_api_key}"
            )
            
            # Enviar con timeout
            with urllib.request.urlopen(url, timeout=10) as response:
                response_text = response.read().decode('utf-8')
                
                if response.status == 200:
                    self.logger.info("Alerta WhatsApp enviada exitosamente")
                    return True
                else:
                    self.logger.error(f"Error WhatsApp: {response.status} - {response_text}")
                    return False
                    
        except Exception as e:
            self.logger.error(f"Error enviando WhatsApp: {e}")
            return False
    
    def _send_log_alert(self, message, priority):
        """Registrar alerta en logs"""
        try:
            if priority == "critical":
                self.logger.critical(f"ALERTA CRITICA: {message}")
            elif priority == "high":
                self.logger.error(f"ALERTA ALTA: {message}")
            elif priority == "medium":
                self.logger.warning(f"ALERTA MEDIA: {message}")
            else:
                self.logger.info(f"ALERTA: {message}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error registrando en log: {e}")
            return False
    
    def _send_email_alert(self, message, alert_data):
        """Enviar alerta por email (placeholder - implementar si es necesario)"""
        # TODO: Implementar envío de email si se requiere
        self.logger.debug("Email no implementado")
        return False
    
    def send_test_alert(self, test_type="info"):
        """Enviar alerta de prueba"""
        test_data = {
            'level': 'HT Crisis' if test_type == "critical" else 'HT1',
            'patient_id': 'TEST_001',
            'sys': 185 if test_type == "critical" else 135,
            'dia': 125 if test_type == "critical" else 85,
            'hr': 95,
            'spo2': 98,
            'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'priority': 'critical' if test_type == "critical" else 'medium'
        }
        
        # Bypass rate limiting para test
        with self.alert_lock:
            self.alert_queue.put(test_data)
            self.logger.info(f"Alerta de prueba enviada: {test_type}")
    
    def get_status(self):
        """Obtener estado del sistema de alertas"""
        return {
            "configured": self.is_configured(),
            "whatsapp_available": bool(self.whatsapp_api_key and self.whatsapp_phone),
            "email_available": self.email_enabled,
            "alerts_sent": self.alerts_sent,
            "alerts_blocked": self.alerts_blocked,
            "queue_size": self.alert_queue.qsize(),
            "last_alert": datetime.fromtimestamp(self.last_alert_time).isoformat() if self.last_alert_time else None,
            "active_histories": len(self.alert_history)
        }
    
    def get_alert_history(self, hours=24):
        """Obtener historial de alertas recientes"""
        current_time = time.time()
        cutoff_time = current_time - (hours * 3600)
        
        recent_alerts = []
        for alert_key, timestamps in self.alert_history.items():
            for timestamp in timestamps:
                if timestamp > cutoff_time:
                    level, patient_id = alert_key.split('_', 1)
                    recent_alerts.append({
                        'level': level,
                        'patient_id': patient_id,
                        'timestamp': datetime.fromtimestamp(timestamp).isoformat(),
                        'time_ago_minutes': int((current_time - timestamp) / 60)
                    })
        
        # Ordenar por tiempo (más reciente primero)
        recent_alerts.sort(key=lambda x: x['timestamp'], reverse=True)
        
        return {
            'period_hours': hours,
            'total_alerts': len(recent_alerts),
            'alerts': recent_alerts[:50]  # Limitar a 50 más recientes
        }
    
    def is_configured(self):
        """Verificar si el sistema está configurado correctamente"""
        return bool(self.whatsapp_api_key and self.whatsapp_phone) or self.email_enabled
    
    def update_configuration(self, config):
        """Actualizar configuración del sistema de alertas"""
        try:
            if 'whatsapp_api_key' in config:
                self.whatsapp_api_key = config['whatsapp_api_key']
            
            if 'whatsapp_phone' in config:
                self.whatsapp_phone = config['whatsapp_phone']
            
            if 'min_alert_interval' in config:
                self.min_alert_interval = int(config['min_alert_interval'])
            
            if 'max_alerts_per_hour' in config:
                self.max_alerts_per_hour = int(config['max_alerts_per_hour'])
            
            self.logger.info("Configuración de alertas actualizada")
            return True
            
        except Exception as e:
            self.logger.error(f"Error actualizando configuración: {e}")
            return False
    
    def add_custom_alert_level(self, level_name, config):
        """Añadir nivel de alerta personalizado"""
        try:
            required_fields = ['priority', 'channels', 'template']
            if not all(field in config for field in required_fields):
                raise ValueError(f"Configuración incompleta. Requeridos: {required_fields}")
            
            valid_channels = ['whatsapp', 'log', 'email']
            invalid_channels = [ch for ch in config['channels'] if ch not in valid_channels]
            if invalid_channels:
                raise ValueError(f"Canales inválidos: {invalid_channels}")
            
            self.alert_levels[level_name] = config
            self.logger.info(f"Nivel de alerta personalizado añadido: {level_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error añadiendo nivel personalizado: {e}")
            return False
    
    def remove_alert_level(self, level_name):
        """Remover nivel de alerta"""
        if level_name in self.alert_levels:
            del self.alert_levels[level_name]
            self.logger.info(f"Nivel de alerta removido: {level_name}")
            return True
        return False
    
    def clear_alert_history(self):
        """Limpiar historial de alertas"""
        with self.alert_lock:
            self.alert_history.clear()
            self.logger.info("Historial de alertas limpiado")
    
    def set_maintenance_mode(self, enabled=True):
        """Activar/desactivar modo mantenimiento (no envía alertas)"""
        if enabled:
            self.maintenance_mode = True
            self.logger.warning("Modo mantenimiento activado - alertas suspendidas")
        else:
            self.maintenance_mode = False
            self.logger.info("Modo mantenimiento desactivado - alertas reanudadas")
    
    def force_send_alert(self, alert_data):
        """Forzar envío de alerta sin verificar rate limiting"""
        try:
            # Preparar datos mínimos si no están completos
            required_fields = {
                'level': 'Test',
                'patient_id': 'Manual',
                'sys': 0,
                'dia': 0,
                'hr': 0,
                'spo2': 0,
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'priority': 'medium'
            }
            
            # Combinar con datos proporcionados
            final_alert_data = {**required_fields, **alert_data}
            
            # Añadir directamente a cola
            self.alert_queue.put(final_alert_data)
            self.logger.info(f"Alerta forzada enviada: {final_alert_data['level']}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error enviando alerta forzada: {e}")
            return False
    
    def get_performance_metrics(self):
        """Obtener métricas detalladas de rendimiento"""
        with self.alert_lock:
            current_time = time.time()
            
            # Calcular alertas por hora en las últimas 24 horas
            recent_alerts = []
            for timestamps in self.alert_history.values():
                for timestamp in timestamps:
                    if current_time - timestamp < 86400:  # 24 horas
                        recent_alerts.append(timestamp)
            
            # Agrupar por horas
            hourly_counts = defaultdict(int)
            for timestamp in recent_alerts:
                hour_key = int(timestamp // 3600)
                hourly_counts[hour_key] += 1
            
            # Calcular estadísticas
            total_recent = len(recent_alerts)
            avg_per_hour = total_recent / 24 if total_recent > 0 else 0
            
            return {
                'total_alerts_sent': self.alerts_sent,
                'total_alerts_blocked': self.alerts_blocked,
                'success_rate': self.alerts_sent / max(self.alerts_sent + self.alerts_blocked, 1),
                'alerts_last_24h': total_recent,
                'avg_alerts_per_hour': round(avg_per_hour, 2),
                'queue_size': self.alert_queue.qsize(),
                'active_rate_limits': len(self.alert_history),
                'hourly_distribution': dict(hourly_counts),
                'uptime_hours': (current_time - getattr(self, 'start_time', current_time)) / 3600
            }
    
    def validate_whatsapp_config(self):
        """Validar configuración de WhatsApp enviando mensaje de prueba"""
        if not (self.whatsapp_api_key and self.whatsapp_phone):
            return {"success": False, "error": "Configuración de WhatsApp incompleta"}
        
        try:
            test_message = "Prueba de configuracion - Sistema de alertas medicas"
            success = self._send_whatsapp_alert(test_message)
            
            if success:
                return {"success": True, "message": "WhatsApp configurado correctamente"}
            else:
                return {"success": False, "error": "Error enviando mensaje de prueba"}
                
        except Exception as e:
            return {"success": False, "error": f"Error validando WhatsApp: {str(e)}"}
    
    def shutdown(self):
        """Apagar sistema de alertas de forma segura"""
        self.logger.info("Iniciando apagado del sistema de alertas...")
        
        # Detener worker thread
        self.should_stop = True
        if self.worker_thread and self.worker_thread.is_alive():
            self.worker_thread.join(timeout=10)
        
        # Procesar alertas pendientes críticas
        critical_alerts = []
        while not self.alert_queue.empty():
            try:
                alert = self.alert_queue.get_nowait()
                if alert.get('priority') == 'critical':
                    critical_alerts.append(alert)
            except queue.Empty:
                break
        
        # Enviar alertas críticas pendientes
        for alert in critical_alerts:
            try:
                self._process_alert(alert)
            except Exception as e:
                self.logger.error(f"Error procesando alerta crítica en shutdown: {e}")
        
        self.logger.info(f"Sistema de alertas cerrado. Alertas críticas procesadas: {len(critical_alerts)}")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        if hasattr(self, 'should_stop'):
            self.should_stop = True
