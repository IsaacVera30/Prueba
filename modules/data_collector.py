# modules/data_collector.py
# RECOLECTOR DE DATOS COMPLETO - Configuración exacta para Render

import os
import csv
import io
import logging
import threading
import numpy as np
from datetime import datetime
from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseUpload

class RealisticTrainingCollector:
    """Recolector para entrenamiento realista con tensiómetro de referencia"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configuración Google Drive - EXACTA PARA TU RENDER
        self.folder_id = "1O3Zmpti5cWXD0X25XpkNgZfuMiRVXw4"  # Tu GOOGLE_DRIVE_FOLDER
        self.csv_filename = "entrenamiento_ml.csv"  # MISMO ARCHIVO EXISTENTE
        self.credentials_path = "/etc/secrets/service_account.json"  # Tu Secret File
        
        # Estados del proceso de entrenamiento
        self.training_active = False
        self.prediction_paused = False  # BLOQUEAR PREDICCIONES
        self.phase = "idle"  # idle, stabilizing, measuring, waiting, ready
        
        # Buffer temporal por fases
        self.sample_buffer = []
        self.phase_start_time = 0
        
        # Configuración de tiempos (en segundos)
        self.timing_config = {
            'stabilization_time': 5,    # 5 segundos estabilización inicial
            'post_measurement_time': 5, # 5 segundos después del tensiómetro
            'min_measurement_time': 10, # Tiempo mínimo de medición
            'max_measurement_time': 60  # Tiempo máximo de medición
        }
        
        # Estadísticas de la sesión
        self.session_stats = {
            'start_time': None,
            'samples_collected': 0,
            'measurement_duration': 0,
            'signal_quality': 0.0,
            'phase_samples': {
                'stabilizing': 0,
                'measuring': 0,
                'waiting': 0
            }
        }
        
        # Lock para thread safety
        self.collection_lock = threading.Lock()
        
        # Servicio Google Drive
        self.drive_service = None
        self._initialize_drive_service()
    
    def _initialize_drive_service(self):
        """Inicializar servicio Google Drive - CONFIGURACIÓN EXACTA DE TU RENDER"""
        try:
            # Verificar que el archivo existe
            if not os.path.exists(self.credentials_path):
                self.logger.error(f"Archivo no encontrado: {self.credentials_path}")
                return False
            
            # Scopes exactos del código viejo que funcionaba
            SCOPES = ["https://www.googleapis.com/auth/drive"]
            
            # Cargar credenciales
            credentials = service_account.Credentials.from_service_account_file(
                self.credentials_path, scopes=SCOPES
            )
            
            # Crear servicio
            self.drive_service = build('drive', 'v3', credentials=credentials)
            
            self.logger.info(f"Google Drive inicializado - Folder: {self.folder_id}")
            self.logger.info(f"Credenciales desde: {self.credentials_path}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error inicializando Google Drive: {e}")
            self.drive_service = None
            return False
    
    def start_training_session(self):
        """Iniciar sesión de entrenamiento - PAUSA PREDICCIONES"""
        with self.collection_lock:
            if self.training_active:
                return {"success": False, "error": "Sesión ya activa"}
            
            # PAUSAR SISTEMA DE PREDICCIÓN
            self.training_active = True
            self.prediction_paused = True
            self.phase = "stabilizing"
            self.phase_start_time = datetime.now().timestamp()
            
            # Limpiar buffers
            self.sample_buffer = []
            
            # Inicializar estadísticas
            self.session_stats = {
                'start_time': datetime.now().isoformat(),
                'samples_collected': 0,
                'measurement_duration': 0,
                'signal_quality': 0.0,
                'phase_samples': {
                    'stabilizing': 0,
                    'measuring': 0,
                    'waiting': 0
                }
            }
        
        self.logger.info("SESIÓN ENTRENAMIENTO INICIADA - PREDICCIONES PAUSADAS")
        
        return {
            "success": True,
            "phase": "stabilizing",
            "message": "Sesión iniciada. Mantenga dedo estable 5 segundos antes de encender tensiómetro",
            "instructions": [
                "1. Mantenga dedo en sensor",
                "2. Espere 5 segundos (estabilización)",
                "3. Encienda tensiómetro en otro brazo",
                "4. Espere que termine medición",
                "5. Mantenga dedo 5 segundos más",
                "6. Detenga captura manualmente"
            ],
            "timestamp": datetime.now().isoformat()
        }
    
    def add_sample(self, sample_data):
        """Añadir muestra durante entrenamiento"""
        if not self.training_active:
            return {"success": False, "error": "Sesión entrenamiento no activa"}
        
        try:
            with self.collection_lock:
                current_time = datetime.now().timestamp()
                elapsed_time = current_time - self.phase_start_time
                
                # Actualizar fase automáticamente
                self._update_phase(elapsed_time)
                
                # Validar calidad de muestra
                quality_result = self._validate_sample_quality(sample_data)
                
                if quality_result['valid']:
                    # Añadir metadatos de fase
                    sample_data.update({
                        'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f'),
                        'phase': self.phase,
                        'elapsed_seconds': round(elapsed_time, 2),
                        'quality_score': quality_result['score']
                    })
                    
                    self.sample_buffer.append(sample_data)
                    self.session_stats['samples_collected'] += 1
                    self.session_stats['phase_samples'][self.phase] += 1
                    
                    # Calcular calidad promedio
                    scores = [s.get('quality_score', 0) for s in self.sample_buffer]
                    self.session_stats['signal_quality'] = np.mean(scores) if scores else 0
                
                # Información de estado actual
                return {
                    "success": True,
                    "phase": self.phase,
                    "elapsed_seconds": round(elapsed_time, 1),
                    "samples_in_phase": self.session_stats['phase_samples'][self.phase],
                    "total_samples": len(self.sample_buffer),
                    "signal_quality": round(self.session_stats['signal_quality'], 2),
                    "sample_valid": quality_result['valid'],
                    "next_instruction": self._get_current_instruction()
                }
                
        except Exception as e:
            self.logger.error(f"Error añadiendo muestra entrenamiento: {e}")
            return {"success": False, "error": str(e)}
    
    def _update_phase(self, elapsed_time):
        """Actualizar fase automáticamente basada en tiempo"""
        if self.phase == "stabilizing" and elapsed_time >= self.timing_config['stabilization_time']:
            self.phase = "ready_for_measurement"
            self.logger.info("Fase: Listo para encender tensiómetro")
    
    def start_measurement_phase(self):
        """Iniciar fase de medición (cuando se enciende tensiómetro)"""
        with self.collection_lock:
            if self.phase != "ready_for_measurement" and self.phase != "stabilizing":
                return {"success": False, "error": f"No se puede iniciar medición desde fase: {self.phase}"}
            
            self.phase = "measuring"
            self.phase_start_time = datetime.now().timestamp()
            
        self.logger.info("FASE MEDICIÓN INICIADA - Tensiómetro encendido")
        
        return {
            "success": True,
            "phase": "measuring",
            "message": "Medición iniciada. Mantenga ambos brazos estables",
            "instruction": "No mover hasta que tensiómetro termine"
        }
    
    def finish_measurement_phase(self):
        """Terminar fase de medición (cuando tensiómetro termina)"""
        with self.collection_lock:
            if self.phase != "measuring":
                return {"success": False, "error": f"No está en fase de medición: {self.phase}"}
            
            # Calcular duración de medición
            measurement_end = datetime.now().timestamp()
            self.session_stats['measurement_duration'] = measurement_end - self.phase_start_time
            
            self.phase = "waiting"
            self.phase_start_time = measurement_end
            
        self.logger.info(f"MEDICIÓN TERMINADA - Duración: {self.session_stats['measurement_duration']:.1f}s")
        
        return {
            "success": True,
            "phase": "waiting",
            "message": f"Medición completada ({self.session_stats['measurement_duration']:.1f}s). Mantenga dedo 5 segundos más",
            "instruction": "Espere 5 segundos adicionales antes de quitar dedo"
        }
    
    def stop_training_session(self):
        """Detener sesión y preparar para guardar datos"""
        with self.collection_lock:
            if not self.training_active:
                return {"success": False, "error": "No hay sesión activa"}
            
            self.training_active = False
            self.phase = "ready_to_save"
            
            # Calcular estadísticas finales
            total_duration = datetime.now().timestamp() - datetime.fromisoformat(self.session_stats['start_time']).timestamp()
            
            session_summary = {
                "total_samples": len(self.sample_buffer),
                "total_duration_seconds": round(total_duration, 1),
                "measurement_duration": round(self.session_stats['measurement_duration'], 1),
                "signal_quality": round(self.session_stats['signal_quality'], 2),
                "phase_breakdown": self.session_stats['phase_samples'],
                "samples_per_second": round(len(self.sample_buffer) / total_duration, 1) if total_duration > 0 else 0
            }
        
        self.logger.info(f"SESIÓN DETENIDA - {session_summary['total_samples']} muestras en {session_summary['total_duration_seconds']}s")
        
        return {
            "success": True,
            "message": "Sesión detenida. Listo para ingresar valores de referencia",
            "session_summary": session_summary,
            "ready_to_save": True
        }
    
    def save_training_data(self, reference_data):
        """Guardar datos de entrenamiento con valores de referencia"""
        try:
            if self.phase != "ready_to_save":
                return {"success": False, "error": "Sesión no lista para guardar"}
            
            if not self.sample_buffer:
                return {"success": False, "error": "No hay muestras para procesar"}
            
            # Validar datos de referencia
            required_refs = ['sys_ref', 'dia_ref', 'hr_ref']
            for field in required_refs:
                if field not in reference_data:
                    return {"success": False, "error": f"Falta campo: {field}"}
            
            # Procesar muestras por fase
            processed_data = self._process_samples_by_phase()
            
            # Crear registro final para CSV - MISMO FORMATO EXISTENTE
            final_record = {
                # FORMATO ORIGINAL del CSV existente
                'hr_promedio_sensor': processed_data['hr_promedio_sensor'],
                'spo2_promedio_sensor': processed_data['spo2_promedio_sensor'],
                'ir_mean_filtrado': processed_data['ir_mean_filtrado'],
                'red_mean_filtrado': processed_data['red_mean_filtrado'],
                'ir_std_filtrado': processed_data['ir_std_filtrado'],
                'red_std_filtrado': processed_data['red_std_filtrado'],
                'sys_ref': float(reference_data['sys_ref']),
                'dia_ref': float(reference_data['dia_ref']),
                'hr_ref': float(reference_data['hr_ref']),
                'timestamp_captura': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            }
            
            # Guardar en Google Drive
            if self.drive_service:
                self._append_to_drive_csv(final_record)
                self.logger.info("Datos entrenamiento guardados en Google Drive")
            else:
                return {"success": False, "error": "Google Drive no disponible"}
            
            # REACTIVAR PREDICCIONES
            self.prediction_paused = False
            self.phase = "idle"
            self.sample_buffer = []
            
            self.logger.info("ENTRENAMIENTO COMPLETADO - PREDICCIONES REACTIVADAS")
            
            return {
                "success": True,
                "message": "Datos guardados exitosamente en Google Drive",
                "record_summary": {
                    "samples_processed": final_record.get('samples_measuring_phase', len(self.sample_buffer)),
                    "measurement_duration": self.session_stats['measurement_duration'],
                    "signal_quality": self.session_stats['signal_quality'],
                    "reference_values": {
                        "sys": final_record['sys_ref'],
                        "dia": final_record['dia_ref'], 
                        "hr": final_record['hr_ref']
                    }
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error guardando datos entrenamiento: {e}")
            # Reactivar predicciones en caso de error
            self.prediction_paused = False
            return {"success": False, "error": str(e)}
    
    def _process_samples_by_phase(self):
        """Procesar muestras separadas por fase"""
        # Separar muestras por fase
        measuring_samples = [s for s in self.sample_buffer if s.get('phase') == 'measuring']
        
        # Si no hay muestras de medición, usar todas las válidas
        if not measuring_samples:
            measuring_samples = [s for s in self.sample_buffer if s.get('quality_score', 0) > 0.7]
        
        if not measuring_samples:
            measuring_samples = self.sample_buffer  # Usar todas como último recurso
        
        # Calcular estadísticas de las muestras de medición
        hr_values = [float(s.get('hr_promedio', 0)) for s in measuring_samples if s.get('hr_promedio')]
        spo2_values = [float(s.get('spo2_sensor', 0)) for s in measuring_samples if s.get('spo2_sensor')]
        ir_values = [float(s.get('ir', 0)) for s in measuring_samples if s.get('ir')]
        red_values = [float(s.get('red', 0)) for s in measuring_samples if s.get('red')]
        
        # Usar promedios ponderados por calidad
        weights = [s.get('quality_score', 1.0) for s in measuring_samples]
        
        return {
            'hr_promedio_sensor': np.average(hr_values, weights=weights) if hr_values else 75,
            'spo2_promedio_sensor': np.average(spo2_values, weights=weights) if spo2_values else 98,
            'ir_mean_filtrado': np.average(ir_values, weights=weights) if ir_values else 1000,
            'red_mean_filtrado': np.average(red_values, weights=weights) if red_values else 800,
            'ir_std_filtrado': np.std(ir_values) if len(ir_values) > 1 else 0,
            'red_std_filtrado': np.std(red_values) if len(red_values) > 1 else 0,
            'samples_measuring_phase': len(measuring_samples),
            'quality_score_measuring': np.mean(weights) if weights else 0
        }
    
    def _get_current_instruction(self):
        """Obtener instrucción actual según la fase"""
        instructions = {
            "stabilizing": "Mantenga dedo estable. NO encienda tensiómetro aún",
            "ready_for_measurement": "Listo. Ahora ENCIENDA el tensiómetro en el otro brazo",
            "measuring": "Medición en curso. Mantenga ambos brazos inmóviles",
            "waiting": "Mantenga dedo unos segundos más, luego detenga captura",
            "ready_to_save": "Ingrese valores del tensiómetro y guarde"
        }
        return instructions.get(self.phase, "Estado desconocido")
    
    def get_training_status(self):
        """Obtener estado completo del entrenamiento"""
        if not self.training_active and self.phase == "idle":
            return {
                "active": False,
                "prediction_system": "ACTIVE",
                "message": "Sistema en modo predicción normal"
            }
        
        current_time = datetime.now().timestamp()
        elapsed = current_time - self.phase_start_time if self.phase_start_time else 0
        
        return {
            "active": self.training_active,
            "prediction_system": "PAUSED" if self.prediction_paused else "ACTIVE",
            "phase": self.phase,
            "elapsed_seconds": round(elapsed, 1),
            "total_samples": len(self.sample_buffer),
            "samples_this_phase": self.session_stats['phase_samples'].get(self.phase, 0),
            "signal_quality": round(self.session_stats['signal_quality'], 2),
            "current_instruction": self._get_current_instruction(),
            "session_stats": self.session_stats
        }
    
    def is_prediction_paused(self):
        """Verificar si las predicciones están pausadas"""
        return self.prediction_paused
    
    def _validate_sample_quality(self, sample_data):
        """Validar calidad de muestra (método simplificado)"""
        try:
            ir = float(sample_data.get('ir', 0))
            red = float(sample_data.get('red', 0))
            
            score = 1.0
            
            # Validar rangos básicos
            if not (5000 <= ir <= 300000):
                score -= 0.4
            if not (3000 <= red <= 200000):
                score -= 0.4
            if red > 0 and not (0.5 <= ir/red <= 5.0):
                score -= 0.2
            
            return {
                'valid': score > 0.6,
                'score': max(0, score),
                'reason': 'Válida' if score > 0.6 else 'Señal fuera de rango'
            }
            
        except:
            return {'valid': False, 'score': 0, 'reason': 'Error procesando'}
    
    def _append_to_drive_csv(self, row_data):
        """Subir a Google Drive - MÉTODO ADAPTADO PARA TU RENDER"""
        try:
            if not self.drive_service:
                raise Exception("Servicio Google Drive no inicializado")
            
            # Buscar archivo existente en tu carpeta
            query = f"name='{self.csv_filename}' and '{self.folder_id}' in parents and trashed=false"
            response = self.drive_service.files().list(
                q=query, 
                spaces='drive', 
                fields='files(id, name)'
            ).execute()
            files = response.get('files', [])
            
            # Preparar datos CSV
            string_io = io.StringIO()
            fieldnames = list(row_data.keys())
            
            if files:
                # Archivo existe - descargar contenido actual
                file_id = files[0]['id']
                self.logger.info(f"Archivo encontrado, ID: {file_id}")
                
                try:
                    existing_file = self.drive_service.files().get_media(fileId=file_id).execute()
                    existing_content = existing_file.decode('utf-8')
                    
                    # Escribir contenido existente
                    string_io.write(existing_content)
                    if not existing_content.strip().endswith('\n'):
                        string_io.write('\n')
                    
                    # Añadir nueva fila
                    string_io.seek(0, io.SEEK_END)
                    csv_writer = csv.writer(string_io)
                    csv_writer.writerow(row_data.values())
                    
                    # MÉTODO DEL CÓDIGO VIEJO: Eliminar archivo anterior
                    self.drive_service.files().delete(fileId=file_id).execute()
                    self.logger.info("Archivo anterior eliminado")
                    
                except Exception as e:
                    self.logger.error(f"Error procesando archivo existente: {e}")
                    # Si falla, crear archivo nuevo
                    string_io = io.StringIO()
                    writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                    writer.writeheader()
                    writer.writerow(row_data)
            else:
                # Archivo nuevo
                self.logger.info("Creando archivo nuevo")
                writer = csv.DictWriter(string_io, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(row_data)
            
            # Subir archivo (MÉTODO EXACTO DEL CÓDIGO VIEJO)
            media = MediaIoBaseUpload(
                io.BytesIO(string_io.getvalue().encode('utf-8')),
                mimetype="text/csv"
            )
            
            file_metadata = {
                "name": self.csv_filename,
                "parents": [self.folder_id]
            }
            
            result = self.drive_service.files().create(
                body=file_metadata,
                media_body=media,
                fields="id"
            ).execute()
            
            self.logger.info(f"Archivo subido exitosamente - ID: {result.get('id')}")
            
        except Exception as e:
            self.logger.error(f"Error subiendo a Google Drive: {e}")
            raise Exception(f"Fallo subida Drive: {e}")

# Mantener clase DataCollector original para compatibilidad
class DataCollector(RealisticTrainingCollector):
    """Alias de compatibilidad"""
    pass
