import os

# Asumimos que ejecutas los scripts desde la raíz del proyecto
RAW_DATA_PATH = "./data/raw/Video_Speech_Actors_01-24" # Dónde están las carpetas Actor_01, etc.
PROCESSED_DATA_DIR = "./data/processed"
METADATA_DIR = "./data/metadata"

INITIAL_CSV_PATH = os.path.join(METADATA_DIR, 'ravdess_initial.csv')
FINAL_CSV_PATH = os.path.join(METADATA_DIR, 'ravdess_final_processed.csv')

# --- PARÁMETROS AUDIO ---
SR = 22050
DURATION = 3 # Segundos
SAMPLES_PER_TRACK = SR * DURATION
N_FFT = 2048
HOP_LENGTH = 512
N_MELS = 128

# --- PARÁMETROS VIDEO ---
IMG_SIZE = 224
FRAMES_PER_VIDEO = 10

import cv2
FACE_CASCADE_PATH = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'

# Diccionario para convertir texto a número
EMOTION_TO_ID = {
    'neutral': 0,
    'calm': 1,
    'happy': 2,
    'sad': 3,
    'angry': 4,
    'fearful': 5,
    'disgust': 6,
    'surprised': 7
}

# Diccionario  (para cuando el modelo prediga '4', saber que es 'angry')
ID_TO_EMOTION = {v: k for k, v in EMOTION_TO_ID.items()}