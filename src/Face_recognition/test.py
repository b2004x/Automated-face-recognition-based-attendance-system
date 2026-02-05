import os
import torch
import random
import numpy as np
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.models import vit_b_16
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from PIL import Image
from torchvision import datasets
import json
from pathlib import Path

transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5],
                         std=[0.5, 0.5, 0.5])
])

model = vit_b_16(weights="IMAGENET1K_V1")
base = Path(__file__).resolve().parent
weight_path = base.parent.parent / "model" / "Face_recognition" / "best_vit_face.pth"
num_classes = 17  # 👈 change this to your number of celebrity classes
model.heads.head = torch.nn.Linear(model.heads.head.in_features, num_classes)
state_dict = torch.load(weight_path, map_location="cpu")
model.load_state_dict(state_dict)
print(" Weights loaded successfully!")

model.eval()
image_path = base.parent.parent / "test" / "Brad_Pitt_2019_by_Glenn_Francis.jpg"



img = Image.open(image_path).convert("RGB")
input_tensor = transforms(img).unsqueeze(0)  # add batch dimension

with torch.no_grad():
    outputs = model(input_tensor)
    probs = torch.nn.functional.softmax(outputs, dim=1)
    pred_class = probs.argmax(dim=1).item()

class_path = base.parent.parent /  "demo" / "back-end" / "class_names.json"

with open(class_path, "r") as f:
    class_names = json.load(f)

predicted_name = class_names[pred_class]
print(f"Predicted Celebrity: {predicted_name}")