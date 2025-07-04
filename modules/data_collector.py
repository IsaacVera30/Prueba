# modules/data_collector.py
# Recolector de datos para entrenamiento ML

import os
import csv
import io
import logging
import threading
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

class DataCollector:
    """Recolector de datos para entrenamiento ML con Google Drive"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración Google Drive
        self.folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID")
        self.csv_filename = "entrenamiento_ml.csv"
        self.service_account_file = "service_account.json"
        
        # Estado del recolector
        self.capture_mode = False
        self.samples_collected = 0
        
        # Buffer temporal para muestras
        self.sample_buffer = []
        self.max_buffer_size = 1000
        
        # Lock para thread safety
        self.collection_lock = threading.Lock()
        
        # Servicio Google Drive
        self.drive_service = None
        
        # Inicializar
        self._initialize_drive_service()
    
    def _initialize_drive_service(self):
        """Inicializar servicio de Google Drive"""
        try:
            if not os.path.exists(self.service_account_file):
                self.logger.warning("Archivo service_account.json no encontrado. Google Drive no disponible.")
                return
            
            if not self.folder_id:
                self.logger.warning("GOOGLE_DRIVE_FOLDER_ID no configurado. Google Drive no disponible.")
                return
            
            # Configurar credenciales
            SCOPES = ['https://www.googleapis.com/auth/drive.file']
            credentials = service_account.Credentials.from_service_account_file(
                self.service_account_file, scopes=SCOPES
            )
            
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.logger.info("Servicio Google Drive inicializado correctamente")
            
        except Exception as e:
            self.logger.error(f"Error inicializando Google Drive: {e}")
            self.drive_service = None
    
    def start_capture(self):
        """Iniciar modo de captura de datos"""
        with self.collection_lock:
            self.capture_mode = True
            self.sample_buffer = []
            self.samples_collected = 0
            
        self.logger.info("Modo de captura de datos iniciado")
        return {"status": "capture_started", "timestamp": datetime.now().isoformat()}
    
    def stop_capture(self):
        """Detener modo de captura"""
        with self.collection_lock:
            was_capturing = self.capture_mode
            self.capture_mode = False
            samples_count = len(self.sample_buffer)
        
        if was_capturing:
            self.logger.info(f"Modo de captura detenido. {samples_count} muestras en buffer")
        
        return {
            "status": "capture_stopped",
            "samples_in_buffer": samples_count,
            "timestamp": datetime.now().isoformat()
        }
    
    def add_sample(self, sample_data):
        """Añadir muestra al buffer de captura"""
        if not self.capture_mode:
            return {"success": False, "error": "Modo captura no activo"}
        
        try:
            with self.collection_lock:
                # Verificar espacio en buffer
                if len(self.sample_buffer) >= self.max_buffer_size:
                    self.logger.warning("Buffer de muestras lleno. Eliminando muestra más antigua.")
                    self.sample_buffer.pop(0)
                
                # Añadir timestamp si no existe
                if 'timestamp_captura' not in sample_data:
                    sample_data['timestamp_captura'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                
                self.sample_buffer.append(sample_data)
                self.samples_collected += 1
            
            self.logger.debug(f"Muestra añadida. Buffer: {len(self.sample_buffer)} muestras")
            
            return {
                "success": True,
                "buffer_size": len(self.sample_buffer),
                "total_collected": self.samples_collected
            }
            
        except Exception as e:
            self.logger.error(f"Error añadiendo muestra: {e}")
            return {"success": False, "error": str(e)}
    
    def save_training_sample(self, sample_data):
        """Guardar muestra de entrenamiento directamente en Google Drive"""
        try:
            if not self.drive_service:
                return {"success": False, "error": "Google Drive no disponible"}
            
            # Validar datos requeridos
            required_fields = [
                'hr_promedio_sensor', 'spo2_promedio_sensor',
                'ir_mean_filtrado', 'red_mean_filtrado',
                'ir_std_filtrado', 'red_std_filtrado',
                'sys_ref', 'dia_ref', 'hr_ref'
            ]
            
            missing_fields = [field for field in required_fields if field not in sample_data]
            if missing_fields:
                return {"success": False, "error": f"Campos faltantes: {missing_fields}"}
            
            # Añadir timestamp si no existe
            if 'timestamp_captura' not in sample_data:
                sample_data['timestamp_captura'] = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            
            # Guardar en Google Drive
            self._append_to_drive_csv(sample_data)
            
            self.logger.info("Muestra de entrenamiento guardada en Google Drive")
            return {"success": True, "message": "Muestra guardada exitosamente"}
            
        except Exception as e:
            self.logger.error(f"Error guardando muestra de entrenamiento: {e}")
            return {"success": False, "error": str(e)}
    
    def save_buffer_to_drive(self, reference_data):
        """Guardar buffer completo con datos de referencia"""
        if not self.capture_mode:
            return {"success": False, "error": "Modo captura no activo"}
        
        if not self.sample_buffer:
            return {"success": False, "error": "Buffer vacío"}
        
        try:
            with self.collection_lock:
                # Calcular promedios del buffer
                avg_sample = self._calculate_buffer_averages()
                
                # Combinar con datos de referencia
                final_sample = {
                    **avg_sample,
                    'sys_ref': reference_data.get('sys_ref'),
                    'dia_ref': reference_data.get('dia_ref'),
                    'hr_ref': reference_data.get('hr_ref'),
                    'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                
                # Guardar en Google Drive
                result = self.save_training_sample(final_sample)
                
                if result['success']:
                    # Limpiar buffer después de guardar
                    buffer_size = len(self.sample_buffer)
                    self.sample_buffer = []
                    self.capture_mode = False
                    
                    self.logger.info(f"Buffer guardado exitosamente. {buffer_size} muestras procesadas")
                    
                    return {
                        "success": True,
                        "samples_processed": buffer_size,
                        "saved_sample": final_sample
                    }
                else:
                    return result
            
        except Exception as e:
            self.logger.error(f"Error guardando buffer: {e}")
            return {"success": False, "error": str(e)}
    
    def _calculate_buffer_averages(self):
        """Calcular promedios de las muestras en buffer"""
        if not self.sample_buffer:
            return {}
        
        import numpy as np
        
        # Extraer valores numéricos
        hr_values = [float(s.get('hr_promedio', 0)) for s in self.sample_buffer if s.get('hr_promedio')]
        spo2_values = [float(s.get('spo2_sensor', 0)) for s in self.sample_buffer if s.get('spo2_sensor')]
        ir_values = [float(s.get('ir', 0)) for s in self.sample_buffer if s.get('ir')]
        red_values = [float(s.get('red', 0)) for s in self.sample_buffer if s.get('red')]
        
        # Calcular estadísticas
        return {
            'hr_promedio_sensor': np.mean(hr_values) if hr_values else 0,
            'spo2_promedio_sensor': np.mean(spo2_values) if spo2_values else 0,
            'ir_mean_filtrado': np.mean(ir_values) if ir_values else 0,
            'red_mean_filtrado': np.mean(red_values) if red_values else 0,
            'ir_std_filtrado': np.std(ir_values) if len(ir_values) > 1 else 0,
            'red_std_filtrado': np.std(red_values) if len(red_values) > 1 else 0
        }
    
    def _append_to_drive_csv(self, row_data):
        """Añadir fila al CSV en Google Drive"""
        try:
            # Buscar archivo existente
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(
                q=query, spaces='drive', fields='files(id, name)'
            ).execute()
            files = response.get('files', [])
            
            # Preparar contenido CSV
            fieldnames = list(row_data.keys())
            string_io = io.StringIO()
            writer = csv.DictWriter(string_io, fieldnames=fieldnames)
            
            if files:
                # Archivo existe - añadir fila
                file_id = files[0].get('id')
                
                # Descargar contenido existente
                existing_file = self.drive_service.files().get_media(fileId=file_id).execute()
                existing_content = existing_file.decode('utf-8')
                
                # Añadir contenido existente
                string_io.write(existing_content)
                if not existing_content.strip().endswith('\n'):
                    string_io.write('\n')
                
                # Añadir nueva fila
                string_io.seek(0, io.SEEK_END)
                csv_writer = csv.writer(string_io)
                csv_writer.writerow(row_data.values())
                
                # Actualizar archivo
                media = MediaIoBaseUpload(
                    io.BytesIO(string_io.getvalue().encode('utf-8')),
                    mimetype='text/csv'
                )
                self.drive_service.files().update(
                    fileId=file_id, media_body=media
                ).execute()
                
            else:
                # Crear nuevo archivo
                writer.writeheader()
                writer.writerow(row_data)
                
                media = MediaIoBaseUpload(
                    io.BytesIO(string_io.getvalue().encode('utf-8')),
                    mimetype='text/csv'
                )
                file_metadata = {
                    'name': self.csv_filename,
                    'parents': [self.folder_id]
                }
                self.drive_service.files().create(
                    body=file_metadata, media_body=media, fields='id'
                ).execute()
            
            self.logger.debug("Fila añadida al CSV en Google Drive")
            
        except Exception as e:
            self.logger.error(f"Error escribiendo en Google Drive: {e}")
            raise
    
    def get_status(self):
        """Obtener estado actual del recolector"""
        return {
            "capture_mode": self.capture_mode,
            "buffer_size": len(self.sample_buffer),
            "samples_collected": self.samples_collected,
            "max_buffer_size": self.max_buffer_size,
            "drive_available": self.drive_service is not None,
            "drive_configured": bool(self.folder_id and os.path.exists(self.service_account_file))
        }
    
    def clear_buffer(self):
        """Limpiar buffer de muestras"""
        with self.collection_lock:
            buffer_size = len(self.sample_buffer)
            self.sample_buffer = []
        
        self.logger.info(f"Buffer limpiado. {buffer_size} muestras descartadas")
        return {"success": True, "samples_cleared": buffer_size}
    
    def export_buffer_to_local(self, filename=None):
        """Exportar buffer a archivo CSV local"""
        if not self.sample_buffer:
            return {"success": False, "error": "Buffer vacío"}
        
        try:
            if not filename:
                filename = f"buffer_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
            
            # Calcular promedios
            avg_sample = self._calculate_buffer_averages()
            
            # Escribir archivo local
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                fieldnames = list(avg_sample.keys())
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(avg_sample)
            
            self.logger.info(f"Buffer exportado a {filename}")
            return {"success": True, "filename": filename, "samples": len(self.sample_buffer)}
            
        except Exception as e:
            self.logger.error(f"Error exportando buffer: {e}")
            return {"success": False, "error": str(e)}
    
    def get_sample_buffer_preview(self, limit=5):
        """Obtener preview de las muestras en buffer"""
        with self.collection_lock:
            preview_samples = self.sample_buffer[-limit:] if self.sample_buffer else []
            
        return {
            "total_samples": len(self.sample_buffer),
            "preview_samples": preview_samples,
            "preview_limit": limit
        }
    
    def validate_sample_data(self, sample_data):
        """Validar que los datos de muestra estén completos y sean válidos"""
        required_fields = [
            'hr_promedio_sensor', 'spo2_promedio_sensor',
            'ir_mean_filtrado', 'red_mean_filtrado',
            'ir_std_filtrado', 'red_std_filtrado'
        ]
        
        validation_result = {
            "valid": True,
            "missing_fields": [],
            "invalid_values": [],
            "warnings": []
        }
        
        # Verificar campos requeridos
        for field in required_fields:
            if field not in sample_data:
                validation_result["missing_fields"].append(field)
                validation_result["valid"] = False
        
        # Verificar rangos válidos
        if validation_result["valid"]:
            ranges = {
                'hr_promedio_sensor': (40, 200),
                'spo2_promedio_sensor': (70, 100),
                'ir_mean_filtrado': (500, 200000),
                'red_mean_filtrado': (300, 150000),
                'ir_std_filtrado': (0, 50000),
                'red_std_filtrado': (0, 30000)
            }
            
            for field, (min_val, max_val) in ranges.items():
                if field in sample_data:
                    try:
                        value = float(sample_data[field])
                        if not (min_val <= value <= max_val):
                            validation_result["invalid_values"].append({
                                "field": field,
                                "value": value,
                                "expected_range": f"{min_val}-{max_val}"
                            })
                            validation_result["valid"] = False
                    except (ValueError, TypeError):
                        validation_result["invalid_values"].append({
                            "field": field,
                            "value": sample_data[field],
                            "error": "No es un número válido"
                        })
                        validation_result["valid"] = False
        
        return validation_result
    
    def get_drive_file_info(self):
        """Obtener información del archivo CSV en Google Drive"""
        if not self.drive_service:
            return {"error": "Google Drive no disponible"}
        
        try:
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(
                q=query, spaces='drive', fields='files(id, name, size, modifiedTime, createdTime)'
            ).execute()
            files = response.get('files', [])
            
            if files:
                file_info = files[0]
                return {
                    "file_exists": True,
                    "file_id": file_info.get('id'),
                    "file_name": file_info.get('name'),
                    "file_size": file_info.get('size'),
                    "created_time": file_info.get('createdTime'),
                    "modified_time": file_info.get('modifiedTime')
                }
            else:
                return {"file_exists": False, "message": "Archivo CSV no encontrado en Drive"}
                
        except Exception as e:
            return {"error": f"Error obteniendo info de Drive: {e}"}
    
    def download_training_data(self, local_filename=None):
        """Descargar datos de entrenamiento desde Google Drive"""
        if not self.drive_service:
            return {"success": False, "error": "Google Drive no disponible"}
        
        try:
            # Buscar archivo
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(
                q=query, spaces='drive', fields='files(id, name)'
            ).execute()
            files = response.get('files', [])
            
            if not files:
                return {"success": False, "error": "Archivo CSV no encontrado en Drive"}
            
            file_id = files[0]['id']
            
            # Descargar contenido
            file_content = self.drive_service.files().get_media(fileId=file_id).execute()
            
            # Guardar localmente
            if not local_filename:
                local_filename = f"downloaded_{self.csv_filename}"
            
            with open(local_filename, 'wb') as f:
                f.write(file_content)
            
            self.logger.info(f"Datos de entrenamiento descargados a {local_filename}")
            return {
                "success": True,
                "filename": local_filename,
                "size_bytes": len(file_content)
            }
            
        except Exception as e:
            self.logger.error(f"Error descargando datos: {e}")
            return {"success": False, "error": str(e)}
