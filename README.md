# Object-Detection-System
with custom dataset for object detection

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T

class CustomObjectDetectionDataset(Dataset):
    """
    A custom dataset for object detection. Expects a directory structure:
    - images/
        - img1.jpg
        - img2.jpg
        - ...
    - annotations/
        - img1.txt  # each line: class_id x_center y_center width height (normalized)
        - img2.txt
        - ...
    """
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.transforms = transforms if transforms else T.ToTensor()

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # Load image
        img_name = self.image_files[idx]
        img_path = os.path.join(self.images_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        # Load annotations
        ann_name = os.path.splitext(img_name)[0] + ".txt"
        ann_path = os.path.join(self.annotations_dir, ann_name)
        boxes = []
        labels = []
        if os.path.exists(ann_path):
            with open(ann_path, 'r') as f:
                for line in f:
                    class_id, x_center, y_center, w, h = map(float, line.strip().split())
                    labels.append(int(class_id))
                    boxes.append([x_center, y_center, w, h])  # normalized format

        # Convert to tensors
        target = {}
        target['boxes'] = torch.tensor(boxes, dtype=torch.float32)
        target['labels'] = torch.tensor(labels, dtype=torch.long)

        # Apply transforms
        image = self.transforms(image)

        return image, target


def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)


def get_dataloader(images_dir,
                   annotations_dir,
                   batch_size=8,
                   shuffle=True,
                   num_workers=4):
    """
    Returns a DataLoader for the custom object detection dataset.
    """
    dataset = CustomObjectDetectionDataset(images_dir, annotations_dir,
                                           transforms=T.Compose([T.ToTensor()]))
    loader = DataLoader(dataset,
                        batch_size=batch_size,
                        shuffle=shuffle,
                        num_workers=num_workers,
                        collate_fn=collate_fn)
    return loader

""" # Example usage:
images_dir = '/path/to/images'
annotations_dir = '/path/to/annotations'
loader = get_dataloader(images_dir, annotations_dir, batch_size=4)
for imgs, targets in loader:
    imgs: list of tensors [3, H, W]
    targets: list of dicts with 'boxes' and 'labels'
    pass"""

