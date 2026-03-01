import pandas as pd
import os
import zipfile

def main():
    # Cargar el CSV original
    csv_path = 'data/metadata/ravdess_final_processed.csv'
    df = pd.read_csv(csv_path)
    
    # Seleccionar solo 30 muestras del conjunto de TEST al azar
    df_demo = df[df['partition'] == 'test'].sample(n=30, random_state=42)
    
    # Guardar un nuevo CSV específico para la demo
    os.makedirs('data/metadata', exist_ok=True)
    demo_csv_path = 'data/metadata/demo_metadata.csv'
    df_demo.to_csv(demo_csv_path, index=False)
    
    # Crear el archivo ZIP respetando la estructura de carpetas
    zip_name = 'demo_data.zip'
    print(f"Empaquetando 30 muestras en {zip_name}...")
    
    with zipfile.ZipFile(zip_name, 'w') as zipf:
        # Añadir el CSV
        zipf.write(demo_csv_path)
        # Añadir los archivos .npy correspondientes
        for _, row in df_demo.iterrows():
            if os.path.exists(row['npy_audio_path']):
                zipf.write(row['npy_audio_path'])
            if os.path.exists(row['npy_video_path']):
                zipf.write(row['npy_video_path'])
                
    print(f"¡Éxito! Archivo {zip_name} creado. Súbelo a Google Drive.")

if __name__ == "__main__":
    main()