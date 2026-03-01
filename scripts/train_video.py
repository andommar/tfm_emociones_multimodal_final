import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd

from src import config
from src.dataset import RAVDESSVideoDataset 
from src.models import VideoHybrid          

# --- CONFIGURACIÓN ---
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Dispositivo de entrenamiento: {device}")

# Limitado por el tamaño del tensor de entrada (B, 10, 3, 224, 224)
BATCH_SIZE = 8  
LEARNING_RATE = 1e-4 
EPOCHS = 10     

# --- PREPARACIÓN DE DATOS ---
df = pd.read_csv(config.FINAL_CSV_PATH)

train_dataset = RAVDESSVideoDataset(df, partition='train')
val_dataset = RAVDESSVideoDataset(df, partition='validation')

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- INICIALIZACIÓN ---
print("Inicializando arquitectura híbrida (ResNet18 + LSTM)...")
model = VideoHybrid(num_classes=8).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- BUCLE DE ENTRENAMIENTO ---
for epoch in range(EPOCHS):
    model.train() 
    running_loss = 0.0
    correct_train = 0
    total_train = 0
    
    for i, (inputs, labels) in enumerate(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total_train += labels.size(0)
        correct_train += (predicted == labels).sum().item()
        
        if i % 10 == 0:
            print(f"   Batch {i}/{len(train_loader)} procesado...")

    train_loss = running_loss / len(train_loader)
    train_acc = 100 * correct_train / total_train

    # --- BUCLE DE VALIDACIÓN ---
    model.eval() 
    val_loss = 0.0
    correct_val = 0
    total_val = 0
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

    val_loss_avg = val_loss / len(val_loader)
    val_acc = 100 * correct_val / total_val

    print(f"Epoch [{epoch+1}/{EPOCHS}] | "
          f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
          f"Val Loss: {val_loss_avg:.4f} | Val Acc: {val_acc:.2f}%")

# --- GUARDADO ---
os.makedirs("models", exist_ok=True)
torch.save(model.state_dict(), "models/video_hybrid_baseline.pth")
print("Modelo guardado en 'models/video_hybrid_baseline.pth'")