import websockets
import asyncio
import yaml
import torch
from vits.model import VITSModel

async def handle_client(websocket, path, model):
    async for message in websocket:
       
        audio_output = model(message) 
        await websocket.send(audio_output) 

async def main(config):
    # Load model
    model = VITSModel(config)
    model.load_state_dict(torch.load(config['final_model_path']))
    model.eval()

    async with websockets.serve(lambda ws, p: handle_client(ws, p, model), "localhost", 8765):
        await asyncio.Future()  # Run forever

if __name__ == "__main__":
    with open("configs/model_config.yaml", "r") as f:
        config = yaml.safe_load(f)
    asyncio.run(main(config))
