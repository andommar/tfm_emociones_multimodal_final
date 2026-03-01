import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class AudioCNN(nn.Module):
    """Arquitectura CNN 2D para el procesamiento de espectrogramas Log-Mel."""
    
    def __init__(self, num_classes=8):
        super(AudioCNN, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # Dimensión esperada para entrada de tamaño (1, 128, 130)
        self.flatten_size = 128 * 8 * 8 
        
        self.fc1 = nn.Linear(self.flatten_size, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        
        return x
    
    def get_features(self, x):
        """Devuelve el embedding de 512 dimensiones previo a la clasificación."""
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.pool4(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1) 
        x = F.relu(self.fc1(x))
        return x
    
class VideoHybrid(nn.Module):
    """Arquitectura híbrida ResNet18 + LSTM para secuencias espacio temporales."""
    
    def __init__(self, num_classes=8, hidden_size=128):
        super(VideoHybrid, self).__init__()
        
        weights = models.ResNet18_Weights.DEFAULT
        self.cnn = models.resnet18(weights=weights)
        
        feature_dim = self.cnn.fc.in_features
        self.cnn.fc = nn.Identity()
        
        self.lstm = nn.LSTM(input_size=feature_dim, hidden_size=hidden_size, 
                            num_layers=1, batch_first=True)
        
        self.fc = nn.Linear(hidden_size, num_classes)
        
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        
        # Aplanado para procesamiento convolucional independiente por frame
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.cnn(c_in)
        
        r_in = c_out.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(r_in)
        
        final_feature = lstm_out[:, -1, :] 
        y = self.fc(final_feature)
        
        return y
    
    def get_features(self, x):
        """Devuelve el estado oculto del último paso de la celda LSTM."""
        batch_size, time_steps, C, H, W = x.size()
        c_in = x.view(batch_size * time_steps, C, H, W)
        c_out = self.cnn(c_in) 
        
        r_in = c_out.view(batch_size, time_steps, -1)
        lstm_out, _ = self.lstm(r_in)
        
        final_feature = lstm_out[:, -1, :] 
        return final_feature


class MultimodalFusion(nn.Module):
    """Fusión a nivel de características (Feature-Level Fusion) de AudioCNN y VideoHybrid."""
    
    def __init__(self, num_classes=8):
        super(MultimodalFusion, self).__init__()
        
        self.audio_branch = AudioCNN()
        self.video_branch = VideoHybrid(hidden_size=128)
        
        # Concatenación: 512 (Audio) + 128 (Vídeo) = 640 dimensiones
        self.fusion_fc = nn.Sequential(
            nn.Linear(512 + 128, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x_audio, x_video):
        feat_audio = self.audio_branch.get_features(x_audio)
        feat_video = self.video_branch.get_features(x_video)
        
        combined = torch.cat((feat_audio, feat_video), dim=1)
        output = self.fusion_fc(combined)
        
        return output