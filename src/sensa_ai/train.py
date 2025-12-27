import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
from src.sensa_ai.model import get_model
from src.sensa_ai.data_loader import SensaDataset, collate_fn, get_transform

def load_params(config_path='params.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0
    pbar = tqdm(data_loader, desc=f'Epoch {epoch}')
    for images, targets in pbar:
        images = list(img.to(device) for img in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        epoch_loss += losses.item()
        pbar.set_postfix({'loss': losses.item()})
    return epoch_loss / len(data_loader)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    args = parser.parse_args()
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        return
    params = load_params(args.config)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    model = get_model(params['num_classes']).to(device)
    optimizer = optim.SGD(model.parameters(), lr=params['learning_rate'], momentum=0.9)
    
    os.makedirs(params['checkpoint_dir'], exist_ok=True)
    # Dummy training loop call
    print("Model initialized. Ready for data.")

if __name__ == '__main__':
    main()
