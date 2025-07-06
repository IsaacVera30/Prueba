# modules/data_collector.py
import os
import csv
import io
import logging
import numpy as np
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

class DataCollector:
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
        self._init_drive()
    
    def _init_drive(self):
        try:
            if not os.path.exists(self.credentials_path):
                self.logger.error("Archivo credentials no encontrado")
                return False
            
            SCOPES = ["https://www.googleapis.com/auth/drive"]
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            self.drive_service = build('drive', 'v3', credentials=credentials)
            self.logger.info("Google Drive OK")
            return True
        except Exception as e:
            self.logger.error(f"Error Drive: {e}")
            return False
    
    # Metodos nuevos
    def start_training_session(self):
        if self.training_active:
            return {"success": False, "error": "Ya activo"}
        
        self.training_active = True
        self.phase = "stabilizing"
        self.sample_buffer = []
        
        self.logger.info("Entrenamiento iniciado")
        return {
            "success": True,
            "phase": "stabilizing",
            "message": "Entrenamiento iniciado"
        }
    
    def add_sample(self, sample_data):
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
            if self.drive_service:
                self._save_to_drive(final_record)
                self.logger.info("Guardado en Drive OK")
            else:
                return {"success": False, "error": "Drive no disponible"}
            
            # Limpiar
            self.phase = "idle"
            self.sample_buffer = []
            
            return {
                "success": True,
                "message": "Guardado exitoso"
            }
            
        except Exception as e:
            self.logger.error(f"Error guardando: {e}")
            return {"success": False, "error": str(e)}
    
    def get_training_status(self):
        return {
            "active": self.training_active,
            "phase": self.phase,
            "total_samples": len(self.sample_buffer),
            "current_instruction": self._get_instruction()
        }
    
    def _process_samples(self):
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
        instructions = {
            "idle": "Inactivo",
            "stabilizing": "Mantener dedo estable",
            "measuring": "Medicion en curso",
            "waiting": "Esperar unos segundos mas",
            "ready_to_save": "Ingresar valores de referencia"
        }
        return instructions.get(self.phase, "Estado desconocido")
    
    def _save_to_drive(self, row_data):
        try:
            # Buscar archivo existente
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(q=query, fields='files(id, name)').execute()
            files = response.get('files', [])
            
            # Preparar CSV
            string_io = io.StringIO()
            fieldnames = list(row_data.keys())
            
            if files:
                # Archivo existe
                file_id = files[0]['id']
                try:
                    existing_file = self.drive_service.files().get_media(fileId=file_id).execute()
                    existing_content = existing_file.decode('utf-8')
                    string_io.write(existing_content)
                    if not existing_content.strip().endswith('\n'):
                        string_io.write('\n')
                    string_io.seek(0, io.SEEK_END)
                    csv_writer = csv.writer(string_io)
                    csv_writer.writerow(row_data.values())
                    self.drive_service.files().delete(fileId=file_id).execute()
                except:
                    string_io = io.StringIO()
                    writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row_data)
            else:
                # Archivo nuevo
                writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)
            
            # Subir archivo
            media = MediaIoBaseUpload(
                io.BytesIO(string_io.getvalue().encode('utf-8')),
                mimetype="text/csv"
            )
            
            file_metadata = {
                "name": self.csv_filename,
                "parents": [self.folder_id]
            }
            
            self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()
            
        except Exception as e:
            raise Exception(f"Error Drive: {e}")
    
    # Metodos legacy para compatibilidad
    def start_capture(self):
        return self.start_training_session()
    
    def stop_capture(self):
        return self.stop_training_session()
    
    def get_status(self):
        return {
            "active": self.training_active,
            "timestamp": datetime.now().isoformat()
        }
    
    def save_training_sample(self, sample_data):
        try:
            self._save_to_drive(sample_data)
            return {"success": True, "message": "Guardado"}
        except Exception as e:
            return {"success": False, "error": str(e)}
