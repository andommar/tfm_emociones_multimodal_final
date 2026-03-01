import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
from src import config

class RAVDESSAudioDataset(Dataset):
    """Dataset para la modalidad unimodal de audio (Log-Mel Spectrograms)."""
    
    def __init__(self, metadata_df, partition='train'):
        self.data = metadata_df[metadata_df['partition'] == partition].reset_index(drop=True)
        self.label_map = config.EMOTION_TO_ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        audio_path = row['npy_audio_path']
        
        try:
            spectrogram = np.load(audio_path)
            spectrogram_tensor = torch.from_numpy(spectrogram).float()
            
            label_id = self.label_map[row['emotion_label']]
            label_tensor = torch.tensor(label_id, dtype=torch.long)
            
            return spectrogram_tensor, label_tensor
            
        except Exception as e:
            print(f"Error loading index {idx} ({audio_path}): {e}")
            # Fallback tensor to maintain batch integrity
            return torch.zeros((1, config.N_MELS, 130)), torch.tensor(0)


class RAVDESSVideoDataset(Dataset):
    """Dataset para la modalidad unimodal de vídeo (Secuencias de fotogramas)."""
    
    def __init__(self, metadata_df, partition='train'):
        self.data = metadata_df[metadata_df['partition'] == partition].reset_index(drop=True)
        self.label_map = config.EMOTION_TO_ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = row['npy_video_path']
        
        try:
            video_frames = np.load(video_path)
            video_tensor = torch.from_numpy(video_frames).float()
            
            # Reordenar dimensiones: (Frames, H, W, C) -> (Frames, C, H, W)
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            
            label_id = self.label_map[row['emotion_label']]
            label_tensor = torch.tensor(label_id, dtype=torch.long)
            
            return video_tensor, label_tensor
            
        except Exception as e:
            print(f"Error loading video index {idx} ({video_path}): {e}")
            return torch.zeros((config.FRAMES_PER_VIDEO, 3, config.IMG_SIZE, config.IMG_SIZE)), torch.tensor(0)
        

class RAVDESSMultimodalDataset(Dataset):
    """Dataset para la fusión multimodal (Audio + Vídeo sincronizados)."""
    
    def __init__(self, metadata_df, partition='train'):
        self.data = metadata_df[metadata_df['partition'] == partition].reset_index(drop=True)
        self.label_map = config.EMOTION_TO_ID

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Procesamiento de la rama de audio
        audio_path = row['npy_audio_path']
        try:
            spectrogram = np.load(audio_path)
            spectrogram_tensor = torch.from_numpy(spectrogram).float()
        except Exception:
            spectrogram_tensor = torch.zeros((1, config.N_MELS, 130))

        # Procesamiento de la rama de vídeo
        video_path = row['npy_video_path']
        try:
            video_frames = np.load(video_path)
            video_tensor = torch.from_numpy(video_frames).float()
            video_tensor = video_tensor.permute(0, 3, 1, 2)
        except Exception:
            video_tensor = torch.zeros((config.FRAMES_PER_VIDEO, 3, config.IMG_SIZE, config.IMG_SIZE))

        # Etiquetado
        label_id = self.label_map[row['emotion_label']]
        label_tensor = torch.tensor(label_id, dtype=torch.long)
        
        return (spectrogram_tensor, video_tensor), label_tensor