import os
import subprocess
import sys

# Install packages
print("Installing packages...")
subprocess.check_call([sys.executable, "-m", "pip", "install", "torch", "torchvision", "matplotlib", "scikit-learn", "kaggle", "-q"])

# Download dataset if not exists
if not os.path.exists('train'):
    print("\nDownloading dataset (2-5 min)...")
    subprocess.run(["kaggle", "datasets", "download", "-d", "birdy654/cifake-real-and-ai-generated-synthetic-images"])
    subprocess.run(["unzip", "-q", "cifake-real-and-ai-generated-synthetic-images.zip"])
    print("Dataset downloaded!")

# Load and display images
print("\nLoading images...")
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = datasets.ImageFolder('train', transform=transform)
test_dataset = datasets.ImageFolder('test', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

print(f"Training images: {len(train_dataset)}")
print(f"Test images: {len(test_dataset)}")
print(f"Classes: {train_dataset.classes}")

images, labels = next(iter(train_loader))
fig, axes = plt.subplots(2, 4, figsize=(12, 6))
axes = axes.flatten()

for idx in range(8):
    img = images[idx].numpy().transpose(1, 2, 0)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = std * img + mean
    img = np.clip(img, 0, 1)
    axes[idx].imshow(img)
    axes[idx].set_title(train_dataset.classes[labels[idx]])
    axes[idx].axis('off')

plt.tight_layout()
plt.savefig('sample_images.png')
print("\n✓ Done! Check sample_images.png")
