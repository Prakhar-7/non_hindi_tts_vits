import numpy as np
import torch
import yaml
import os
from torch.utils.data import DataLoader, Dataset
from vits.model import VITSModel  # Assuming VITS is already cloned and model is available

class AudioDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.npy')]
    
    def __len__(self):
        return len(self.files)
    
    def __getitem__(self, idx):
        data = np.load(self.files[idx])
        return torch.tensor(data, dtype=torch.float32)

def train_model(config):
    
    dataset = AudioDataset(os.path.join(config['data_path'], "processed"))
    dataloader = DataLoader(dataset, batch_size=config['batch_size'], shuffle=True)

    model = VITSModel(config)
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])

    for epoch in range(config['epochs']):
        for data in dataloader:
            optimizer.zero_grad()
            output = model(data)
            loss = ... 
            loss.backward()
            optimizer.step()
        print(f"Epoch [{epoch+1}/{config['epochs']}], Loss: {loss.item()}")

        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(config['checkpoint_path'], f"model_epoch_{epoch+1}.pth")
            torch.save(model.state_dict(), checkpoint_path)
            print(f"Checkpoint saved at {checkpoint_path}")

if __name__ == "__main__":
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_model(config)
