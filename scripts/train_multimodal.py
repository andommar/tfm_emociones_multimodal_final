import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from src import config
from src.dataset import RAVDESSMultimodalDataset
from src.models import MultimodalFusion

# --- CONFIGURACIÓN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

BATCH_SIZE = 8
LEARNING_RATE = 1e-4 
EPOCHS = 15

# --- PREPARACIÓN DE DATOS ---
df = pd.read_csv(config.FINAL_CSV_PATH)

train_ds = RAVDESSMultimodalDataset(df, partition='train')
val_ds = RAVDESSMultimodalDataset(df, partition='validation')

train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

# --- INICIALIZACIÓN Y TRANSFER LEARNING ---
model = MultimodalFusion(num_classes=8).to(device)

try:
    model.audio_branch.load_state_dict(torch.load("models/audio_cnn_baseline.pth", map_location=device))
    model.video_branch.load_state_dict(torch.load("models/video_hybrid_baseline.pth", map_location=device))
    print("Pesos unimodales base cargados exitosamente.")
except Exception as e:
    print(f"Aviso: Entrenamiento desde cero. No se pudieron cargar pesos pre-entrenados: {e}")

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- BUCLE DE ENTRENAMIENTO ---
for epoch in range(EPOCHS):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for i, ((audio_in, video_in), labels) in enumerate(train_loader):
        audio_in, video_in = audio_in.to(device), video_in.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        outputs = model(audio_in, video_in)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    train_acc = 100 * correct / total
    train_loss = running_loss / len(train_loader)

    # --- BUCLE DE VALIDACIÓN ---
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    
    with torch.no_grad():
        for (audio_in, video_in), labels in val_loader:
            audio_in, video_in = audio_in.to(device), video_in.to(device)
            labels = labels.to(device)

            outputs = model(audio_in, video_in)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            val_total += labels.size(0)
            val_correct += (predicted == labels).sum().item()
            
    val_acc = 100 * val_correct / val_total
    
    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss/len(val_loader):.4f} | Val Acc: {val_acc:.2f}%")

# --- GUARDADO ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/multimodal_fusion.pth")
print("Modelo multimodal guardado en 'models/multimodal_fusion.pth'")