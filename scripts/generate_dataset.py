import os
import glob
import pandas as pd
import numpy as np
from tqdm import tqdm

from src import config
from src import preprocessors

def generate_initial_metadata(root_path):
    """Generates initial metadata dataframe from RAW directory structure."""
    print(f"Escaneando directorio: {root_path}")
    file_list = []
    
    search_pattern = os.path.join(root_path, "**", "*.mp4")
    files = glob.glob(search_pattern, recursive=True)
    
    if not files:
        print(f"Error: No se encontraron archivos .mp4 en {root_path}")
        return pd.DataFrame()

    for filepath in files:
        filename = os.path.basename(filepath)
        parts = filename.split('.')[0].split('-')
        
        if len(parts) != 7: 
            continue
        
        file_list.append({
            'path': filepath,
            'filename': filename,
            'modality': parts[0],
            'emotion': parts[2],
            'actor': int(parts[6])
        })
        
    return pd.DataFrame(file_list)

def assign_partition(actor_id):
    """
    Asigna particiones basadas en ID de actor (Leave-One-Speaker-Group-Out).
    Test: Actores 21-24. Val: Actores 19-20. Train: Actores 1-18.
    """
    test_actors = [21, 22, 23, 24]
    val_actors = [19, 20]
    
    if actor_id in test_actors:
        return 'test'
    elif actor_id in val_actors:
        return 'validation'
    else:
        return 'train'

def main():
    os.makedirs(config.METADATA_DIR, exist_ok=True)
    audio_out_dir = os.path.join(config.PROCESSED_DATA_DIR, 'audio')
    video_out_dir = os.path.join(config.PROCESSED_DATA_DIR, 'video')
    
    os.makedirs(audio_out_dir, exist_ok=True)
    os.makedirs(video_out_dir, exist_ok=True)

    df = generate_initial_metadata(config.RAW_DATA_PATH)
    if df.empty: 
        return
        
    df.to_csv(config.INITIAL_CSV_PATH, index=False)

    # Filtrar solo archivos audiovisuales (Modalidad '01')
    df_filtered = df[df['modality'].astype(str) == '01'].copy()
    print(f"Total archivos AV a procesar: {len(df_filtered)}")

    processed_audio_paths = []
    processed_video_paths = []
    valid_indices = []

    print("Iniciando extracción de características...")
    for idx, row in tqdm(df_filtered.iterrows(), total=df_filtered.shape[0]):
        original_path = row['path']
        base_name = row['filename'].split('.')[0]

        npy_audio_path = os.path.join(audio_out_dir, f"{base_name}.npy")
        npy_video_path = os.path.join(video_out_dir, f"{base_name}.npy")

        # Skip si ya fueron preprocesados
        if os.path.exists(npy_audio_path) and os.path.exists(npy_video_path):
            processed_audio_paths.append(npy_audio_path)
            processed_video_paths.append(npy_video_path)
            valid_indices.append(idx)
            continue

        try:
            audio_data = preprocessors.process_audio_file(original_path)
            video_data = preprocessors.process_video_file(original_path)

            if audio_data is not None and video_data is not None:
                np.save(npy_audio_path, audio_data.astype(np.float32))
                np.save(npy_video_path, video_data.astype(np.float32))
                
                processed_audio_paths.append(npy_audio_path)
                processed_video_paths.append(npy_video_path)
                valid_indices.append(idx)

        except Exception as e:
            print(f"Error procesando {base_name}: {e}")

    df_final = df_filtered.loc[valid_indices].copy()
    
    df_final['npy_audio_path'] = processed_audio_paths
    df_final['npy_video_path'] = processed_video_paths
    
    emotion_map = {
        '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad', 
        '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
    }
    
    df_final['emotion_label'] = df_final['emotion'].astype(str).str.zfill(2).map(emotion_map)
    df_final['partition'] = df_final['actor'].apply(assign_partition)

    df_final.to_csv(config.FINAL_CSV_PATH, index=False)
    print(f"Dataset generado exitosamente en: {config.FINAL_CSV_PATH}")

if __name__ == "__main__":
    main()