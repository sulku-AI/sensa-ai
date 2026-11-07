"""
Custom PyTorch Dataset for Sensa AI
Loads images and XML annotations (Pascal VOC format)
"""

import os
import torch
import xml.etree.ElementTree as ET
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T

class SensaDataset(Dataset):
    def __init__(self, root_dir, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        
        # Resim dosyalarini bul ve sirala (Sadece jpg/png)
        self.images = sorted([f for f in os.listdir(root_dir) if f.endswith(('.jpg', '.jpeg', '.png'))])
        
        # Etiket haritasi (MakeSense'de yazdigin etiketler buraya)
        self.label_to_idx = {
            'background': 0,
            'blood': 1,
            'gore': 2
            # 'violence': 3 (Kullanmadigimiz icin kapali)
        }
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # 1. Resmi Yukle
        img_name = self.images[idx]
        img_path = os.path.join(self.root_dir, img_name)
        image = Image.open(img_path).convert('RGB')
        
        # 2. XML Dosyasini Bul ve Oku
        # Resim isminin uzantisini silip .xml ekliyoruz
        xml_name = os.path.splitext(img_name)[0] + '.xml'
        xml_path = os.path.join(self.root_dir, xml_name)
        
        boxes = []
        labels = []
        
        if os.path.exists(xml_path):
            tree = ET.parse(xml_path)
            root = tree.getroot()
            
            for obj in root.findall('object'):
                name = obj.find('name').text.lower()
                if name in self.label_to_idx:
                    label = self.label_to_idx[name]
                    
                    bndbox = obj.find('bndbox')
                    xmin = float(bndbox.find('xmin').text)
                    ymin = float(bndbox.find('ymin').text)
                    xmax = float(bndbox.find('xmax').text)
                    ymax = float(bndbox.find('ymax').text)
                    
                    boxes.append([xmin, ymin, xmax, ymax])
                    labels.append(label)
        
        # 3. Tensorlara Cevir
        target = {}
        if len(boxes) > 0:
            target['boxes'] = torch.as_tensor(boxes, dtype=torch.float32)
            target['labels'] = torch.as_tensor(labels, dtype=torch.int64)
            area = (target['boxes'][:, 3] - target['boxes'][:, 1]) * (target['boxes'][:, 2] - target['boxes'][:, 0])
            target['area'] = area
        else:
            # Eger etiket yoksa (Temiz resim)
            target['boxes'] = torch.zeros((0, 4), dtype=torch.float32)
            target['labels'] = torch.zeros((0,), dtype=torch.int64)
            target['area'] = torch.zeros((0,), dtype=torch.float32)
            
        target['image_id'] = torch.tensor([idx])
        target['iscrowd'] = torch.zeros((len(boxes),), dtype=torch.int64)
        
        if self.transforms:
            image = self.transforms(image)
        
        return image, target

def collate_fn(batch):
    return tuple(zip(*batch))

def get_transform(train=True):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.ColorJitter(brightness=0.2, contrast=0.2))
    return T.Compose(transforms)