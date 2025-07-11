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
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.folder_id = os.environ.get("GOOGLE_DRIVE_FOLDER_ID", "1O3Zmpti5cWXD0X25XpkNgZfuMiRVXw4")
        self.csv_filename = "entrenamiento_ml.csv"
        self.credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", "/etc/secrets/service_account.json")
        self.training_active = False
        self.sample_buffer = []
        self.phase = "idle"
        self.drive_service = None
        self.drive_connected = False
        self.last_connection_attempt = 0
        self.connection_retry_interval = 60
        self.total_saved = 0
        self.save_errors = 0
        self.last_save_time = 0
        self.save_lock = threading.Lock()
        self._init_drive()
    
    def _init_drive(self):
        try:
            if not os.path.exists(self.credentials_path):
                return False
            
            SCOPES = ["https://www.googleapis.com/auth/drive"]
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            self.drive_service = build('drive', 'v3', credentials=credentials)
            
            if self._test_drive_connection():
                self.drive_connected = True
                return True
            else:
                return False
                
        except Exception as e:
            self.drive_connected = False
            return False
    
    def _test_drive_connection(self):
        try:
            response = self.drive_service.files().list(pageSize=1).execute()
            
            try:
                folder_info = self.drive_service.files().get(
                    fileId=self.folder_id,
                    fields='id, name, mimeType'
                ).execute()
                return True
            except HttpError:
                return False
                
        except Exception:
            return False
    
    def save_training_sample(self, sample_data):
        with self.save_lock:
            try:
                if not self._validate_sample_data(sample_data):
                    return {"success": False, "error": "Datos inválidos"}
                
                if not self.drive_connected:
                    if not self._init_drive():
                        return {"success": False, "error": "Google Drive no disponible"}
                
                for attempt in range(3):
                    try:
                        result = self._save_to_drive_simple(sample_data)
                        if result.get('success'):
                            self.total_saved += 1
                            self.last_save_time = time.time()
                            return {"success": True, "message": "Guardado exitoso"}
                        
                        if attempt < 2:
                            time.sleep(1)
                            self._init_drive()
                        
                    except Exception as e:
                        if attempt < 2:
                            time.sleep(1)
                            self._init_drive()
                        continue
                
                self.save_errors += 1
                return {"success": False, "error": "Error después de 3 intentos"}
                
            except Exception as e:
                self.save_errors += 1
                return {"success": False, "error": str(e)}
    
    def _validate_sample_data(self, sample_data):
        try:
            required_fields = [
                'hr_promedio_sensor', 'spo2_promedio_sensor',
                'ir_mean_filtrado', 'red_mean_filtrado',
                'ir_std_filtrado', 'red_std_filtrado',
                'sys_ref', 'dia_ref', 'hr_ref'
            ]
            
            for field in required_fields:
                if field not in sample_data:
                    return False
            
            sys_ref = float(sample_data['sys_ref'])
            dia_ref = float(sample_data['dia_ref'])
            hr_ref = float(sample_data['hr_ref'])
            
            if not (70 <= sys_ref <= 250) or not (40 <= dia_ref <= 150) or not (40 <= hr_ref <= 200):
                return False
            
            if dia_ref >= sys_ref:
                return False
            
            ir_mean = float(sample_data['ir_mean_filtrado'])
            red_mean = float(sample_data['red_mean_filtrado'])
            
            if ir_mean <= 0 or red_mean <= 0:
                return False
            
            return True
            
        except Exception:
            return False
    
    def _save_to_drive_simple(self, row_data):
        try:
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(q=query, fields='files(id, name)', pageSize=1).execute()
            files = response.get('files', [])
            
            fieldnames = [
                'hr_promedio_sensor', 'spo2_promedio_sensor',
                'ir_mean_filtrado', 'red_mean_filtrado',
                'ir_std_filtrado', 'red_std_filtrado',
                'sys_ref', 'dia_ref', 'hr_ref',
                'timestamp_captura'
            ]
            
            csv_content = ""
            
            if files:
                file_id = files[0]['id']
                try:
                    existing_data = self.drive_service.files().get_media(fileId=file_id).execute()
                    csv_content = existing_data.decode('utf-8')
                    self.drive_service.files().delete(fileId=file_id).execute()
                except Exception:
                    csv_content = ""
            
            if not csv_content.strip():
                output = io.StringIO()
                writer = csv.writer(output)
                writer.writerow(fieldnames)
                csv_content = output.getvalue()
                output.close()
            
            output = io.StringIO()
            output.write(csv_content)
            if not csv_content.endswith('\n'):
                output.write('\n')
            
            writer = csv.writer(output)
            new_row = [row_data.get(field, '') for field in fieldnames]
            writer.writerow(new_row)
            
            final_content = output.getvalue()
            output.close()
            
            media = MediaIoBaseUpload(
                io.BytesIO(final_content.encode('utf-8')),
                mimetype="text/csv"
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
            
            return {
                "success": True,
                "file_id": uploaded_file.get('id'),
                "file_name": uploaded_file.get('name')
            }
            
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def get_status(self):
        return {
            "drive_connected": self.drive_connected,
            "folder_id": self.folder_id,
            "csv_filename": self.csv_filename,
            "total_saved": self.total_saved,
            "save_errors": self.save_errors,
            "last_save": datetime.fromtimestamp(self.last_save_time).isoformat() if self.last_save_time else None,
            "timestamp": datetime.now().isoformat()
        }
    
    def test_drive_connection(self):
        try:
            result = self._test_drive_connection()
            if result:
                return {"success": True, "message": "Conexión exitosa"}
            else:
                return {"success": False, "error": "Conexión fallida"}
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def force_reconnect(self):
        self.drive_connected = False
        self.drive_service = None
        return self._init_drive()
    
    def clear_errors(self):
        self.save_errors = 0
