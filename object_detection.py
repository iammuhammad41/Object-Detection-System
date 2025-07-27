import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as T
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

class CustomObjectDetectionDataset(Dataset):
    """
    Directory structure:
      D:\PythonProjects\datasets\object_dataset\
        ├── images/
        │     ├── img1.jpg
        │     ├── img2.jpg
        │     └── ...
        └── annotations/
              ├── img1.txt    # each line: class_id x_center y_center width height (normalized)
              ├── img2.txt
              └── ...
    """
    def __init__(self, images_dir, annotations_dir, transforms=None):
        self.images_dir = images_dir
        self.annotations_dir = annotations_dir
        self.image_files = sorted(os.listdir(images_dir))
        self.transforms = transforms if transforms is not None else T.ToTensor()

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
        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        # Apply transforms
        image = self.transforms(image)

        return image, target

def collate_fn(batch):
    images, targets = list(zip(*batch))
    return list(images), list(targets)

def get_dataloader(images_dir, annotations_dir, batch_size=4, shuffle=True, num_workers=2):
    dataset = CustomObjectDetectionDataset(
        images_dir, 
        annotations_dir,
        transforms=T.Compose([T.ToTensor()])
    )
    return DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        num_workers=num_workers, 
        collate_fn=collate_fn
    )

def get_model(num_classes):
    # Load a model pre-trained on COCO
    model = fasterrcnn_resnet50_fpn(pretrained=True)
    # Classifier with a new one for the number of classes (including background)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

if __name__ == "__main__":
    # Dataset Path
    images_dir = r"D:\PythonProjects\datasets\object_dataset\images"
    annotations_dir = r"D:\PythonProjects\datasets\object_dataset\annotations"

    # DataLoader
    dataloader = get_dataloader(images_dir, annotations_dir, batch_size=2)

    # No of object classes
    num_classes = 2

    # model
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = get_model(num_classes).to(device)

    # Optimizer
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

    # Training loop (one epoch example)
    model.train()
    for images, targets in dataloader:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        print(f"Loss: {losses.item():.4f}")
