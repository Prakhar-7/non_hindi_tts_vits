import torch
import yaml
from vits.model import VITSModel
from src.train import AudioDataset
from torch.utils.data import DataLoader
import numpy as np

def evaluate_model(config):
    
    model = VITSModel(config)
    model.load_state_dict(torch.load(config['final_model_path']))
    model.eval()

    test_dataset = AudioDataset(os.path.join(config['data_path'], "processed"))
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    with torch.no_grad():
        for data in test_loader:
            output = model(data)
           
            mse = np.mean((data.numpy() - output.numpy())**2)
            print(f"Mean Squared Error: {mse}")

if __name__ == "__main__":
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    evaluate_model(config)
