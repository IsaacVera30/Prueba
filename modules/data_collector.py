# modules/data_collector.py
# Recolector de datos especializado para entrenamiento ML - SISTEMA CORREGIDO

import os
import csv
import io
import logging
import numpy as np
import time
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload
from googleapiclient.errors import HttpError
import threading

class DataCollector:
    """Recolector de datos especializado para entrenamiento ML con manejo robusto"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuracion Google Drive
        self.folder_id = "1O3Zmpti5cWXD0X25XpkNgZfuMiRVXw4"
        self.csv_filename = "entrenamiento_ml.csv"
        self.credentials_path = "/etc/secrets/service_account.json"
        
        # Estados
        self.training_active = False
        self.sample_buffer = []
        self.phase = "idle"
        
        # Configuracion Drive
        self.drive_service = None
        self.drive_connected = False
        self.last_connection_attempt = 0
        self.connection_retry_interval = 300  # 5 minutos
        
        # Métricas y estadísticas
        self.total_saved = 0
        self.save_errors = 0
        self.last_save_time = 0
        
        # Lock para thread safety
        self.save_lock = threading.Lock()
        
        # Inicializar Drive
        self._init_drive()
    
    def _init_drive(self):
        """Inicializar conexión con Google Drive con manejo robusto de errores"""
        try:
            if not os.path.exists(self.credentials_path):
                self.logger.error(f"Archivo credentials no encontrado en: {self.credentials_path}")
                self.logger.warning("Data Collector funcionando sin Google Drive")
                return False
            
            SCOPES = ["https://www.googleapis.com/auth/drive"]
            
            # Cargar credenciales
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            
            # Crear servicio Drive
            self.drive_service = build('drive', 'v3', credentials=credentials)
            
            # Probar conexión
            if self._test_drive_connection():
                self.drive_connected = True
                self.logger.info("Google Drive inicializado correctamente")
                return True
            else:
                self.logger.error("Falló la prueba de conexión con Google Drive")
                return False
                
        except Exception as e:
            self.logger.error(f"Error inicializando Google Drive: {e}")
            self.drive_connected = False
            return False
    
    def _test_drive_connection(self):
        """Probar conexión con Google Drive"""
        try:
            # Verificar que la carpeta existe
            query = f"id='{self.folder_id}' and trashed=false"
            response = self.drive_service.files().list(q=query, fields='files(id, name)').execute()
            files = response.get('files', [])
            
            if files:
                self.logger.info(f"Carpeta Drive encontrada: {files[0].get('name', 'Sin nombre')}")
                return True
            else:
                self.logger.error(f"Carpeta Drive no encontrada con ID: {self.folder_id}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error probando conexión Drive: {e}")
            return False
    
    def _retry_drive_connection(self):
        """Reintentar conexión con Google Drive si ha pasado tiempo suficiente"""
        current_time = time.time()
        if (current_time - self.last_connection_attempt) > self.connection_retry_interval:
            self.last_connection_attempt = current_time
            self.logger.info("Reintentando conexión con Google Drive...")
            self._init_drive()
    
    def start_training_session(self):
        """Iniciar sesión de entrenamiento - LEGACY COMPATIBILITY"""
        if self.training_active:
            return {"success": False, "error": "Ya activo"}
        
        self.training_active = True
        self.phase = "stabilizing"
        self.sample_buffer = []
        
        self.logger.info("Entrenamiento iniciado (legacy)")
        return {
            "success": True,
            "phase": "stabilizing",
            "message": "Entrenamiento iniciado"
        }
    
    def add_sample(self, sample_data):
        """Añadir muestra - LEGACY COMPATIBILITY"""
        if not self.training_active:
            return {"success": False, "error": "No activo"}
        
        try:
            sample_data.update({
                'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                'phase': self.phase
            })
            
            self.sample_buffer.append(sample_data)
            
            return {
                "success": True,
                "phase": self.phase,
                "total_samples": len(self.sample_buffer),
                "next_instruction": "Mantener dedo estable"
            }
        except Exception as e:
            self.logger.error(f"Error añadiendo muestra: {e}")
            return {"success": False, "error": str(e)}
    
    def stop_training_session(self):
        """Detener sesión de entrenamiento - LEGACY COMPATIBILITY"""
        if not self.training_active:
            return {"success": False, "error": "No activo"}
        
        self.training_active = False
        self.phase = "ready_to_save"
        
        return {
            "success": True,
            "message": "Sesion detenida",
            "session_summary": {
                "total_samples": len(self.sample_buffer)
            },
            "ready_to_save": True
        }
    
    def save_training_data(self, ref_data):
        """Guardar datos de entrenamiento con referencias - LEGACY COMPATIBILITY"""
        try:
            if self.phase != "ready_to_save":
                return {"success": False, "error": "No listo para guardar"}
            
            if not self.sample_buffer:
                return {"success": False, "error": "No hay muestras"}
            
            # Procesar muestras
            processed_data = self._process_samples()
            
            # Crear registro final
            final_record = {
                'hr_promedio_sensor': processed_data['hr_promedio'],
                'spo2_promedio_sensor': processed_data['spo2_promedio'],
                'ir_mean_filtrado': processed_data['ir_mean'],
                'red_mean_filtrado': processed_data['red_mean'],
                'ir_std_filtrado': processed_data['ir_std'],
                'red_std_filtrado': processed_data['red_std'],
                'sys_ref': float(ref_data['sys_ref']),
                'dia_ref': float(ref_data['dia_ref']),
                'hr_ref': float(ref_data['hr_ref']),
                'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Guardar en Drive
            result = self.save_training_sample(final_record)
            
            if result.get('success'):
                # Limpiar después de guardar exitosamente
                self.phase = "idle"
                self.sample_buffer = []
                
                return {
                    "success": True,
                    "message": "Guardado exitoso"
                }
            else:
                return result
            
        except Exception as e:
            self.logger.error(f"Error guardando datos: {e}")
            return {"success": False, "error": str(e)}
    
    def save_training_sample(self, sample_data):
        """
        Método principal para guardar muestras de entrenamiento
        Compatible con el nuevo sistema de buffers separados
        """
        with self.save_lock:
            try:
                # Validar datos de entrada
                if not self._validate_sample_data(sample_data):
                    return {"success": False, "error": "Datos de muestra inválidos"}
                
                # Verificar conexión Drive
                if not self.drive_connected:
                    self._retry_drive_connection()
                
                if not self.drive_connected:
                    self.logger.error("Google Drive no disponible")
                    self.save_errors += 1
                    return {"success": False, "error": "Google Drive no disponible"}
                
                # Guardar en Drive con reintentos
                result = self._save_to_drive_with_retry(sample_data)
                
                if result.get('success'):
                    self.total_saved += 1
                    self.last_save_time = time.time()
                    self.logger.info("Muestra de entrenamiento guardada exitosamente")
                    return {"success": True, "message": "Muestra guardada en Google Drive"}
                else:
                    self.save_errors += 1
                    return result
                
            except Exception as e:
                self.logger.error(f"Error crítico guardando muestra: {e}")
                self.save_errors += 1
                return {"success": False, "error": f"Error crítico: {str(e)}"}
    
    def _validate_sample_data(self, sample_data):
        """Validar que los datos de la muestra sean correctos"""
        try:
            required_fields = [
                'hr_promedio_sensor', 'spo2_promedio_sensor',
                'ir_mean_filtrado', 'red_mean_filtrado',
                'ir_std_filtrado', 'red_std_filtrado',
                'sys_ref', 'dia_ref', 'hr_ref'
            ]
            
            # Verificar que todos los campos estén presentes
            missing_fields = [field for field in required_fields if field not in sample_data]
            if missing_fields:
                self.logger.error(f"Campos faltantes: {missing_fields}")
                return False
            
            # Validar rangos de valores de referencia
            sys_ref = float(sample_data['sys_ref'])
            dia_ref = float(sample_data['dia_ref'])
            hr_ref = float(sample_data['hr_ref'])
            
            if not (70 <= sys_ref <= 250):
                self.logger.error(f"SYS fuera de rango: {sys_ref}")
                return False
            
            if not (40 <= dia_ref <= 150):
                self.logger.error(f"DIA fuera de rango: {dia_ref}")
                return False
            
            if not (40 <= hr_ref <= 200):
                self.logger.error(f"HR fuera de rango: {hr_ref}")
                return False
            
            if dia_ref >= sys_ref:
                self.logger.error(f"DIA >= SYS: {dia_ref} >= {sys_ref}")
                return False
            
            # Validar datos del sensor
            ir_mean = float(sample_data['ir_mean_filtrado'])
            red_mean = float(sample_data['red_mean_filtrado'])
            
            if ir_mean <= 0 or red_mean <= 0:
                self.logger.error(f"Señales inválidas - IR: {ir_mean}, RED: {red_mean}")
                return False
            
            self.logger.debug("Validación de datos exitosa")
            return True
            
        except Exception as e:
            self.logger.error(f"Error validando datos: {e}")
            return False
    
    def _save_to_drive_with_retry(self, sample_data, max_retries=3):
        """Guardar en Drive con sistema de reintentos"""
        for attempt in range(max_retries):
            try:
                self.logger.info(f"Guardando en Drive (intento {attempt + 1}/{max_retries})")
                result = self._save_to_drive(sample_data)
                
                if result.get('success'):
                    return result
                else:
                    self.logger.warning(f"Intento {attempt + 1} falló: {result.get('error')}")
                    
            except Exception as e:
                self.logger.error(f"Error en intento {attempt + 1}: {e}")
                
                # Si es error de conexión, reintentar conexión Drive
                if "connection" in str(e).lower() or "timeout" in str(e).lower():
                    self.drive_connected = False
                    self._init_drive()
            
            # Esperar antes del siguiente intento (excepto en el último)
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)  # Backoff exponencial
        
        return {"success": False, "error": f"Falló después de {max_retries} intentos"}
    
    def _save_to_drive(self, row_data):
        """Guardar datos en Google Drive"""
        try:
            # Buscar archivo existente
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(q=query, fields='files(id, name)').execute()
            files = response.get('files', [])
            
            # Preparar contenido CSV
            string_io = io.StringIO()
            fieldnames = [
                'hr_promedio_sensor', 'spo2_promedio_sensor',
                'ir_mean_filtrado', 'red_mean_filtrado',
                'ir_std_filtrado', 'red_std_filtrado',
                'sys_ref', 'dia_ref', 'hr_ref',
                'timestamp_captura'
            ]
            
            if files:
                # Archivo existe - descargar y añadir nueva fila
                file_id = files[0]['id']
                try:
                    # Descargar contenido existente
                    existing_file = self.drive_service.files().get_media(fileId=file_id).execute()
                    existing_content = existing_file.decode('utf-8')
                    
                    # Escribir contenido existente
                    string_io.write(existing_content)
                    if not existing_content.strip().endswith('\n'):
                        string_io.write('\n')
                    
                    # Posicionarse al final y añadir nueva fila
                    string_io.seek(0, io.SEEK_END)
                    csv_writer = csv.writer(string_io)
                    
                    # Preparar fila en el orden correcto
                    row_values = [row_data.get(field, '') for field in fieldnames]
                    csv_writer.writerow(row_values)
                    
                    # Eliminar archivo anterior
                    self.drive_service.files().delete(fileId=file_id).execute()
                    self.logger.debug("Archivo anterior eliminado")
                    
                except HttpError as e:
                    if e.resp.status == 404:
                        # Archivo no encontrado, crear nuevo
                        self.logger.warning("Archivo no encontrado, creando nuevo")
                        string_io = io.StringIO()
                        writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                        writer.writeheader()
                        writer.writerow(row_data)
                    else:
                        raise
                        
            else:
                # Archivo nuevo - crear con header
                writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)
                self.logger.info("Creando nuevo archivo CSV")
            
            # Subir archivo a Drive
            csv_content = string_io.getvalue()
            media = MediaIoBaseUpload(
                io.BytesIO(csv_content.encode('utf-8')),
                mimetype="text/csv",
                resumable=True
            )
            
            file_metadata = {
                "name": self.csv_filename,
                "parents": [self.folder_id]
            }
            
            uploaded_file = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id, name, size"
            ).execute()
            
            self.logger.info(f"Archivo guardado: {uploaded_file.get('name')} "
                           f"(ID: {uploaded_file.get('id')}, "
                           f"Tamaño: {uploaded_file.get('size', 0)} bytes)")
            
            return {
                "success": True,
                "file_id": uploaded_file.get('id'),
                "file_name": uploaded_file.get('name'),
                "file_size": uploaded_file.get('size', 0)
            }
            
        except HttpError as e:
            error_msg = f"Error HTTP Drive: {e.resp.status} - {e.content.decode()}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
            
        except Exception as e:
            error_msg = f"Error guardando en Drive: {str(e)}"
            self.logger.error(error_msg)
            return {"success": False, "error": error_msg}
    
    def get_training_status(self):
        """Obtener estado del entrenamiento - LEGACY COMPATIBILITY"""
        return {
            "active": self.training_active,
            "phase": self.phase,
            "total_samples": len(self.sample_buffer),
            "current_instruction": self._get_instruction()
        }
    
    def _process_samples(self):
        """Procesar muestras del buffer - LEGACY COMPATIBILITY"""
        hr_values = [float(s.get('hr_promedio', 0)) for s in self.sample_buffer if s.get('hr_promedio')]
        spo2_values = [float(s.get('spo2_sensor', 0)) for s in self.sample_buffer if s.get('spo2_sensor')]
        ir_values = [float(s.get('ir', 0)) for s in self.sample_buffer if s.get('ir')]
        red_values = [float(s.get('red', 0)) for s in self.sample_buffer if s.get('red')]
        
        return {
            'hr_promedio': np.mean(hr_values) if hr_values else 75,
            'spo2_promedio': np.mean(spo2_values) if spo2_values else 98,
            'ir_mean': np.mean(ir_values) if ir_values else 1000,
            'red_mean': np.mean(red_values) if red_values else 800,
            'ir_std': np.std(ir_values) if len(ir_values) > 1 else 0,
            'red_std': np.std(red_values) if len(red_values) > 1 else 0
        }
    
    def _get_instruction(self):
        """Obtener instrucción actual - LEGACY COMPATIBILITY"""
        instructions = {
            "idle": "Inactivo",
            "stabilizing": "Mantener dedo estable",
            "measuring": "Medicion en curso",
            "waiting": "Esperar unos segundos mas",
            "ready_to_save": "Ingresar valores de referencia"
        }
        return instructions.get(self.phase, "Estado desconocido")
    
    def get_status(self):
        """Obtener estado completo del data collector"""
        return {
            "drive_connected": self.drive_connected,
            "training_active": self.training_active,
            "phase": self.phase,
            "samples_in_buffer": len(self.sample_buffer),
            "total_saved": self.total_saved,
            "save_errors": self.save_errors,
            "last_save": datetime.fromtimestamp(self.last_save_time).isoformat() if self.last_save_time else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def get_file_info(self):
        """Obtener información del archivo CSV en Drive"""
        if not self.drive_connected:
            return {"success": False, "error": "Drive no conectado"}
        
        try:
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(
                q=query, 
                fields='files(id, name, size, modifiedTime, createdTime)'
            ).execute()
            files = response.get('files', [])
            
            if files:
                file_info = files[0]
                return {
                    "success": True,
                    "exists": True,
                    "file_id": file_info.get('id'),
                    "name": file_info.get('name'),
                    "size_bytes": int(file_info.get('size', 0)),
                    "size_mb": round(int(file_info.get('size', 0)) / (1024 * 1024), 2),
                    "created": file_info.get('createdTime'),
                    "modified": file_info.get('modifiedTime')
                }
            else:
                return {
                    "success": True,
                    "exists": False,
                    "message": "Archivo no existe, se creará en el primer guardado"
                }
                
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def test_drive_connection(self):
        """Probar conexión con Google Drive manualmente"""
        try:
            result = self._test_drive_connection()
            if result:
                return {
                    "success": True,
                    "message": "Conexión con Google Drive exitosa",
                    "folder_id": self.folder_id,
                    "file_name": self.csv_filename
                }
            else:
                return {
                    "success": False,
                    "error": "Falló la conexión con Google Drive"
                }
        except Exception as e:
            return {
                "success": False,
                "error": f"Error probando conexión: {str(e)}"
            }
    
    def reset_connection(self):
        """Resetear conexión con Google Drive"""
        self.drive_connected = False
        self.drive_service = None
        self.last_connection_attempt = 0
        
        result = self._init_drive()
        
        return {
            "success": result,
            "message": "Conexión reseteada y reinicializada" if result else "Falló el reset de conexión"
        }
    
    def get_performance_metrics(self):
        """Obtener métricas de rendimiento detalladas"""
        total_operations = self.total_saved + self.save_errors
        success_rate = (self.total_saved / total_operations * 100) if total_operations > 0 else 0
        
        return {
            "total_saved": self.total_saved,
            "save_errors": self.save_errors,
            "success_rate": round(success_rate, 2),
            "drive_connected": self.drive_connected,
            "last_save_time": self.last_save_time,
            "connection_retries": getattr(self, 'connection_retries', 0),
            "avg_save_time": getattr(self, 'avg_save_time', 0),
            "folder_id": self.folder_id,
            "csv_filename": self.csv_filename
        }
    
    def clear_buffer(self):
        """Limpiar buffer de muestras"""
        self.sample_buffer = []
        self.phase = "idle"
        self.training_active = False
        self.logger.info("Buffer de muestras limpiado")
        
        return {"success": True, "message": "Buffer limpiado"}
    
    def shutdown(self):
        """Apagar data collector de forma segura"""
        try:
            self.logger.info("Cerrando Data Collector...")
            
            # Guardar muestras pendientes si las hay
            if self.sample_buffer and self.phase == "ready_to_save":
                self.logger.warning("Hay muestras pendientes en buffer al cerrar")
            
            # Limpiar estado
            self.clear_buffer()
            self.drive_connected = False
            self.drive_service = None
            
            self.logger.info("Data Collector cerrado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error cerrando Data Collector: {e}")
    
    def __del__(self):
        """Limpieza al destruir el objeto"""
        try:
            if hasattr(self, 'sample_buffer'):
                self.shutdown()
        except:
            pass
