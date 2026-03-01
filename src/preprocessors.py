import numpy as np
import librosa
import cv2
import os
from src import config # Importamos nuestra propia configuración


# Inicializar detector de rostros de OpenCV (Haar Cascades)
face_cascade = cv2.CascadeClassifier(config.FACE_CASCADE_PATH)

# ===========================
# FUNCIONES DE PREPROCESAMIENTO DE AUDIO
# ===========================
def process_audio_file(file_path):
    """
    Toma un archivo de audio (o video extrayendo su pista de audio),
    calcula su Espectrograma de Mel en escala logarítmica (dB) y 
    devuelve un tensor listo para alimentar la red neuronal.
    """
    try:
        # Cargar el audio y forzar la misma tasa de muestreo
        y, sr = librosa.load(file_path, sr=config.SR, mono=True)
        length = len(y)
        
        # --- Normalización de la longitud temporal ---
        # Las CNNs esperan entradas de tamaño fijo.
        # Si el audio es más largo de lo esperado lo cortamos
        if length > config.SAMPLES_PER_TRACK:
            y = y[:config.SAMPLES_PER_TRACK]
        # Si es más corto, rellenamos con ceros (silencio) al final (padding)
        else:
            padding = config.SAMPLES_PER_TRACK - length
            y = np.pad(y, (0, padding), 'constant')
            
        # Convertir la onda de audio a un Espectrograma de Mel
        mel = librosa.feature.melspectrogram(
            y=y, sr=sr, n_fft=config.N_FFT, 
            hop_length=config.HOP_LENGTH, n_mels=config.N_MELS
        )
        
        # Convertir la energía a decibelios (escala logarítmica)
        log_mel = librosa.power_to_db(mel, ref=np.max)
        
        # Formatear para PyTorch
        # PyTorch espera tensores en formato (Canales, Altura, Anchura).
        # Nuestro espectrograma es (Frecuencias, Tiempo), así que le añadimos
        # una dimensión extra al principio para simular el canal (como una imagen en escala de grises).
        # Resultado final: (1, 128, 130).
        return log_mel[np.newaxis, ...] 

    except Exception as e:
        print(f"Error procesando audio {os.path.basename(file_path)}: {e}")
        return None

# ===========================
# FUNCIONES DE PREPROCESAMIENTO DE VIDEO
# ===========================
def crop_center(frame):
    """
    Si el detector facial falla
    recortamos el centro del fotograma asumiendo que el actor está ahí
    """
    h, w, _ = frame.shape
    c_y, c_x = h // 2, w // 2 # Encontrar el centro exacto
    m = min(h, w) // 2        # Tomar el lado más corto para hacer un recorte cuadrado
    return frame[c_y - m:c_y + m, c_x - m:c_x + m]

def process_video_file(file_path):
    """
    Abre un video extrae una cantidad fija de fotogramas espaciados uniformemente
    detecta el rostro en cada uno y normaliza los píxeles
    devuelve un array 4D que representa la secuencia temporal.
    """
    cap = cv2.VideoCapture(file_path)
    if not cap.isOpened(): 
        return None
    
    # Averiguar cuántos fotogramas tiene el video en total
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0: 
        return None

    # --- Mmuestreo uniforme ---
    # En lugar de coger los N primeros frames, calculamos N índices espaciados
    # para capturar toda la evolución de la emoción (inicio, clímax, fin).
    indices = np.linspace(0, total_frames-1, config.FRAMES_PER_VIDEO, dtype=int)
    frames_buffer = []
    
    current_frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret: 
            break # Fin del video
        
        # Si el fotograma actual coincide con uno de los índices que calculamos antes
        if current_frame_idx in indices:
            
            # --- Fase de detección facial ---
            # Haar Cascades necesita la imagen en blanco y negro
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(30,30))
            
            # Si hemos detectado al menos una cara...
            if len(faces) > 0:
                # Buscamos la cara con mayor área (ancho * alto) por si hay falsos positivos en el fondo
                x, y, w, h = max(faces, key=lambda i: i[2]*i[3])
                # Recortamos solo la región de la cara
                face_img = frame[y:y+h, x:x+w]
            else:
                # Si el detector falla aplicamos recorte central
                face_img = crop_center(frame)
                
            # Redimensionar para que todos los recortes tengan el mismo tamaño (ej. 224x224)
            face_resized = cv2.resize(face_img, (config.IMG_SIZE, config.IMG_SIZE))
            # OpenCV lee en BGR por defecto lo pasamos a RGB 
            face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
            
            # Normalizamos los valores de los píxeles de [0, 255] a un rango continuo de [0.0, 1.0]
            frames_buffer.append(face_rgb / 255.0)
            
            # Si ya tenemos todos los fotogramas que queríamos, dejamos de leer el video
            if len(frames_buffer) == config.FRAMES_PER_VIDEO: 
                break
            
        current_frame_idx += 1
            
    # Liberar el recurso del video
    cap.release()
    
    # Si por algún problema el video tenía menos frames útiles
    # hasta alcanzar la cantidad deseada (config.FRAMES_PER_VIDEO). 
    # similar al padding de audio pero replicando la última imagen.
    if len(frames_buffer) > 0:
        while len(frames_buffer) < config.FRAMES_PER_VIDEO:
            frames_buffer.append(frames_buffer[-1])
    else:
         return None # No se pudo extraer ningún frame válido

    # Convertir a array de numpy. 
    # La forma final es (N_Frames, Alto, Ancho, Canales) -> ej: (10, 224, 224, 3)
    # Luego en el Dataset de PyTorch tendremos que permutar esto a (10, 3, 224, 224)
    return np.array(frames_buffer, dtype=np.float32)