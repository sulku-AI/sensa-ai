#!/usr/bin/env python3
"""
Sensa AI Project Setup Script
Fixed: Handles root directory files correctly.
"""

import os
import json

def create_directory(path):
    os.makedirs(path, exist_ok=True)
    print(f"✓ Created directory: {path}")

def write_file(path, content):
    # FIX: Only try to create directory if path has a parent folder
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    
    with open(path, 'w') as f:
        f.write(content)
    print(f"✓ Created file: {path}")

def setup_project():
    print("=" * 60)
    print("Setting up Sensa AI Project Structure...")
    print("=" * 60)
    
    # 1. Create Directories
    directories = [
        '.github/workflows', 'data/raw', 'data/processed', 
        'models', 'notebooks', 'src/sensa_ai', 'tests'
    ]
    for directory in directories:
        create_directory(directory)
    
    # Create .gitkeep for empty folders
    for gitkeep_dir in ['data/raw', 'data/processed', 'models']:
        write_file(os.path.join(gitkeep_dir, '.gitkeep'), '')
    
    # 2. params.yaml
    params_content = """# Sensa AI Training Parameters
epochs: 50
batch_size: 8
learning_rate: 0.005
num_classes: 3
train_data_dir: data/raw/train
val_data_dir: data/raw/val
checkpoint_dir: models
model_name: sensa_ai_model.pth
device: auto
num_workers: 4
pin_memory: true
save_frequency: 10
"""
    write_file('params.yaml', params_content)
    
    # 3. init.py
    write_file('src/sensa_ai/__init__.py', '__version__ = "0.1.0"')
    
    # 4. model.py
    model_content = """import torch
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

def get_model_summary(model):
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")
    return total_params
"""
    write_file('src/sensa_ai/model.py', model_content)
    
    # 5. data_loader.py
    data_loader_content = """import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SensaDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.annotations = []
        if os.path.exists(root_dir):
            for file in os.listdir(root_dir):
                if file.endswith('.json'):
                    with open(os.path.join(root_dir, file), 'r') as f:
                        self.annotations.append(json.load(f))
        self.label_to_idx = {'background': 0, 'blood': 1, 'violence': 2, 'gore': 2}
        
    def __len__(self):
        return len(self.annotations)
    
    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        img_path = os.path.join(self.root_dir, annotation['file'])
        image = Image.open(img_path).convert('RGB')
        boxes, labels = [], []
        for obj in annotation.get('objects', []):
            boxes.append(obj['bbox'])
            labels.append(self.label_to_idx.get(obj['label'].lower(), 0))
        
        target = {}
        target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32) if boxes else torch.zeros((0, 4), dtype=torch.float32)
        target['labels'] = torch.as_tensor(labels, dtype=torch.int64) if labels else torch.zeros((0,), dtype=torch.int64)
        target['image_id'] = torch.tensor([idx])
        
        if self.transforms:
            image = self.transforms(image)
        else:
            image = T.ToTensor()(image)
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train=True):
    transforms = [T.ToTensor()]
    if train:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    return T.Compose(transforms)
"""
    write_file('src/sensa_ai/data_loader.py', data_loader_content)

    # 6. train.py
    train_content = """import os
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
"""
    write_file('src/sensa_ai/train.py', train_content)
    
    # 7. Extras
    write_file('requirements.txt', "torch>=2.0.0\ntorchvision>=0.15.0\npyyaml\nopencv-python\ntqdm\nPillow")
    write_file('.gitignore', "__pycache__/\n*.py[cod]\nvenv/\n.env\ndata/raw/*\ndata/processed/*\nmodels/*\n!data/raw/.gitkeep\n!models/.gitkeep\n")
    
    readme_content = """# Sensa AI
    Real-time sensitive content filtering.
    """
    write_file('README.md', readme_content)
    write_file('notebooks/colab_train.ipynb', '{"cells":[], "metadata":{}}')
    write_file('tests/__init__.py', '')

    print("\n" + "=" * 60)
    print("✅ Sensa AI project setup complete!")
    print("=" * 60)

if __name__ == '__main__':
    setup_project()
    git add .
git commit -m "feat: complete project structure setup"
git push