# Object Detection System

This repository contains a PyTorch-based custom object detection pipeline using Faster R‑CNN. It includes:

* A `CustomObjectDetectionDataset` for loading images and YOLO-style annotations
* A collate function and DataLoader wrapper
* A script to initialize and train a Faster R‑CNN model on your own dataset

---

## 🗂️ Project Structure

```
D:\PythonProjects\datasets\object_dataset
├── images/          # .jpg image files
│     ├── img1.jpg
│     ├── img2.jpg
│     └── ...
└── annotations/     # .txt annotation files
      ├── img1.txt   # lines: class_id x_center y_center width height (normalized)
      ├── img2.txt
      └── ...

object_detection.py  # Main script defining Dataset, DataLoader, model, and training loop
README.md            # This file
```

## 🔧 Requirements

* Python 3.8+
* PyTorch 1.7+
* torchvision 0.8+
* PIL (Pillow)

Install dependencies via pip:

```bash
pip install torch torchvision pillow
```

## ⚙️ Configuration

* **COCO Dataset path**: Update the paths in `object_detection.py`:

  ```python
  images_dir = r"D:\PythonProjects\datasets\object_dataset\images"
  annotations_dir = r"D:\PythonProjects\datasets\object_dataset\annotations"
  ```
* **Number of classes**:
  Set `num_classes` (including the background) in the `__main__` block.

## ▶️ Usage

1. Place your images (`.jpg`) in the `images/` folder and matching `.txt` files in `annotations/`.
2. Ensure the annotation format: each line is `class_id x_center y_center width height`, all values normalized to \[0,1].
3. Run training:

   ```bash
   python object_detection.py
   ```
4. Monitor the console for training losses.

## 📈 Training Loop

The script:

* Instantiates a pretrained Faster R‑CNN with a ResNet-50 FPN backbone
* Replaces the final box predictor for your `num_classes`
* Uses an SGD optimizer (lr=0.005, momentum=0.9, weight\_decay=0.0005)
* Performs one epoch over the dataset (extendable)

You can customize:

* Learning rate and optimizer in the `optimizer` section
* Number of epochs by wrapping the training loop
* Add validation, checkpointing, or learning-rate schedulers

## 🛠️ Customization Tips

* **Transforms**: Modify `transforms=T.Compose([...])` for data augmentation
* **Backbone**: Swap `resnet50_fpn` for other detection backbones
* **Batch Size & Workers**: Adjust `batch_size` and `num_workers` in `get_dataloader`

---

Happy Training! 🚀
