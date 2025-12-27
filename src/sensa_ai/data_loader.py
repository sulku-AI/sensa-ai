import os
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
