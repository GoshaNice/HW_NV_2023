from src.preprocessing.melspectrogram import MelSpectrogram, MelSpectrogramConfig
import torchaudio
import torch
import numpy as np

wav2spec = MelSpectrogram(MelSpectrogramConfig())

def load_audio(path):
    audio_tensor, sr = torchaudio.load(path)
    audio_tensor = audio_tensor[0:1, :]  # remove all channels but the first
    target_sr = 22050
    if sr != target_sr:
        audio_tensor = torchaudio.functional.resample(audio_tensor, sr, target_sr)
    return audio_tensor

def process_wave(audio_tensor_wave):
    with torch.no_grad():
        audio_tensor_spec = wav2spec(audio_tensor_wave)
        return audio_tensor_wave, audio_tensor_spec
    

for i in range(1, 4):
    path = f"test_data_folder/audio_{i}.wav"
    audio_tensor_wave = load_audio(path)
    audio_tensor_wave, audio_tensor_spec = process_wave(audio_tensor_wave)
    audio_spec = audio_tensor_spec.cpu().detach().numpy()
    np.save(f"test_data_folder/mel_{i}.npy", audio_spec)
