import os
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
from torch.utils.data import DataLoader

from src import config
from src.dataset import RAVDESSMultimodalDataset
from src.models import MultimodalFusion

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluación en dispositivo: {device}")

    df = pd.read_csv(config.FINAL_CSV_PATH)
    test_ds = RAVDESSMultimodalDataset(df, partition='test')
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    model = MultimodalFusion(num_classes=8).to(device)
    model_path = "models/multimodal_fusion.pth"

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Archivo de pesos no encontrado: {model_path}")

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for (audio_in, video_in), labels in test_loader:
            audio_in, video_in = audio_in.to(device), video_in.to(device)
            labels = labels.to(device)
            
            outputs = model(audio_in, video_in)
            _, predicted = torch.max(outputs, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    emotion_names = [config.ID_TO_EMOTION[i] for i in range(8)]
    report = classification_report(all_labels, all_preds, target_names=emotion_names, digits=4)
    
    print("\n--- REPORTE DE CLASIFICACIÓN (TEST SET) ---\n")
    print(report)

    os.makedirs("reports", exist_ok=True)
    with open("reports/classification_report.txt", "w") as f:
        f.write(report)

    cm = confusion_matrix(all_labels, all_preds)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=emotion_names, yticklabels=emotion_names)
    plt.xlabel('Predicción')
    plt.ylabel('Ground Truth')
    plt.title('Matriz de Confusión: Fusión Multimodal')
    plt.tight_layout()
    plt.savefig("reports/confusion_matrix.png")
    
    print("Métricas exportadas a la carpeta 'reports/'.")

if __name__ == "__main__":
    main()