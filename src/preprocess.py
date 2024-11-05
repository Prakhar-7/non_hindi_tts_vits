import os
import librosa
import numpy as np
import yaml

def normalize_audio(file_path, target_sample_rate=22050):
    y, sr = librosa.load(file_path, sr=target_sample_rate)
    return y

def preprocess_data(config):
    data_path = config['data_path']
    processed_path = os.path.join(data_path, "processed")
    os.makedirs(processed_path, exist_ok=True)
    
    for file in os.listdir(data_path):
        if file.endswith('.wav'):
            audio_path = os.path.join(data_path, file)
            y = normalize_audio(audio_path)
            np.save(os.path.join(processed_path, file.replace('.wav', '.npy')), y)
    print("Data preprocessing complete.")

if __name__ == "__main__":
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    preprocess_data(config)
