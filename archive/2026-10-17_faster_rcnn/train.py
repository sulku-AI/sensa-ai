"""
Training script for Sensa AI (Real Logic)
"""
import os
import sys
import argparse
import yaml
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.sensa_ai.model import get_model, get_model_summary
from src.sensa_ai.data_loader import SensaDataset, collate_fn, get_transform


def load_params(config_path='params.yaml'):
    with open(config_path, 'r') as f:
        params = yaml.safe_load(f)
    return params


def train_one_epoch(model, optimizer, data_loader, device, epoch):
    model.train()
    epoch_loss = 0.0
    
    # Progress bar
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch}')
    
    for images, targets in progress_bar:
        # Move to device
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]
        
        # Forward pass
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())
        
        # Backward pass
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()
        
        # Update statistics
        epoch_loss += losses.item()
        progress_bar.set_postfix({'loss': losses.item()})
    
    return epoch_loss / len(data_loader) if len(data_loader) > 0 else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='params.yaml')
    args = parser.parse_args()
    
    # Load params
    if not os.path.exists(args.config):
        print(f"HATA: {args.config} bulunamadi!")
        return
    params = load_params(args.config)
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"🔥 Using device: {device}")
    
    # Model Setup
    model = get_model(num_classes=params['num_classes'])
    model.to(device)
    
    # Optimizer
    optimizer = optim.SGD(
        model.parameters(),
        lr=params['learning_rate'],
        momentum=0.9,
        weight_decay=0.0005
    )
    
    # Datasets
    print("Loading data...")
    train_dataset = SensaDataset(
        root_dir=params['train_data_dir'],
        transforms=get_transform(train=True)
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=params['batch_size'],
        shuffle=True,
        num_workers=2,
        collate_fn=collate_fn
    )
    
    print(f"Training samples: {len(train_dataset)}")
    
    # Training Loop
    os.makedirs(params['checkpoint_dir'], exist_ok=True)
    print("🚀 Training Started!")
    
    for epoch in range(params['epochs']):
        loss = train_one_epoch(model, optimizer, train_loader, device, epoch + 1)
        print(f"Epoch {epoch+1}/{params['epochs']} - Loss: {loss:.4f}")
        
        # Save checkpoint
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), os.path.join(params['checkpoint_dir'], f"{params['model_name']}_ep{epoch+1}.pth"))
            
    # Final Save
    torch.save(model.state_dict(), os.path.join(params['checkpoint_dir'], params['model_name']))
    print("✅ Training Complete!")


if __name__ == '__main__':
    main()