import os
import cv2
import torch
import numpy as np
from models.scnn import SCNN
from utils.dataset import CULaneDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torchvision.utils import save_image
from matplotlib import pyplot as plt

# ==== Settings ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
IMG_PATH = "/path/to/CULane/driver_161_90frame/image/05181633_0376.jpg"
CHECKPOINT = "checkpoints/scnn_epoch20.pth"
NUM_CLASSES = 5

# ==== Transform ====
predict_transform = A.Compose([
    A.Resize(288, 800),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==== Load Image ====
orig = cv2.imread(IMG_PATH)
orig = cv2.cvtColor(orig, cv2.COLOR_BGR2RGB)
transformed = predict_transform(image=orig)['image'].unsqueeze(0).to(DEVICE)

# ==== Load Model ====
model = SCNN(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
model.load_state_dict(torch.load(CHECKPOINT, map_location=DEVICE))
model.eval()

# ==== Inference ====
with torch.no_grad():
    output = model(transformed)
    preds = torch.argmax(output.squeeze(), dim=0).cpu().numpy()

# ==== Visualize ====
def overlay_lanes(image, mask):
    color_map = np.array([
        [0, 0, 0],        # background
        [255, 0, 0],      # lane 1
        [0, 255, 0],      # lane 2
        [0, 0, 255],      # lane 3
        [255, 255, 0]     # lane 4
    ])
    mask_color = color_map[mask]
    overlay = cv2.addWeighted(image, 0.6, mask_color.astype(np.uint8), 0.4, 0)
    return overlay

overlay_img = overlay_lanes(orig, preds)

plt.imshow(overlay_img)
plt.axis('off')
plt.title("Predicted Lanes")
plt.show()
