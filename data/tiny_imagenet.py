import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNetDataset(Dataset):
    def __init__(self, root_path, split='train', transform=None):
        """
        Args:
            root_path (str): Path to the extracted 'tiny-imagenet-200' directory.
            split (str): 'train', 'val', or 'test'.
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.root_path = root_path
        self.split = split
        self.transform = transform
        self.image_paths = []
        self.labels = []
        
        # 1. Load class IDs (wnids)
        wnids_path = os.path.join(root_path, 'wnids.txt')
        with open(wnids_path, 'r') as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        # Create mapping from class ID to integer index
        self.label_to_idx = {cls: i for i, cls in enumerate(self.classes)}
        
        # 2. Parse the specific split
        if split == 'train':
            train_dir = os.path.join(root_path, 'train')
            for cls in self.classes:
                cls_dir = os.path.join(train_dir, cls, 'images')
                for img_name in os.listdir(cls_dir):
                    if img_name.endswith('.jpg'):
                        self.image_paths.append(os.path.join(cls_dir, img_name))
                        self.labels.append(self.label_to_idx[cls])
                        
        elif split == 'val':
            val_dir = os.path.join(root_path, 'val')
            val_annotations_path = os.path.join(val_dir, 'val_annotations.txt')
            with open(val_annotations_path, 'r') as f:
                for line in f.readlines():
                    parts = line.strip().split('\t')
                    img_name = parts[0]
                    cls = parts[1]
                    self.image_paths.append(os.path.join(val_dir, 'images', img_name))
                    self.labels.append(self.label_to_idx[cls])
                    
        elif split == 'test':
            # Note: Test set has no labels provided by standard Tiny ImageNet
            test_dir = os.path.join(root_path, 'test', 'images')
            for img_name in os.listdir(test_dir):
                if img_name.endswith('.JPEG'):
                    self.image_paths.append(os.path.join(test_dir, img_name))
                    self.labels.append(-1) # Placeholder for unknown labels

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        
        # Convert to RGB is CRITICAL. Some Tiny ImageNet images are grayscale, 
        # which will crash CNNs expecting 3 channels.
        image = Image.open(img_path).convert('RGB')
        label_idx = self.labels[idx]
        
        if self.transform:
            image = self.transform(image)
            
        return image, label_idx