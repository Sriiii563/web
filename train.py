import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from models.scnn import SCNN
from utils.dataset import CULaneDataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

# ==== Config ====
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 5  # Background + 4 lanes
BATCH_SIZE = 8
EPOCHS = 20
LEARNING_RATE = 1e-4
SAVE_PATH = "checkpoints"
os.makedirs(SAVE_PATH, exist_ok=True)

# ==== Data Paths ====
CULANE_ROOT = "/path/to/CULane"
IMG_DIR = os.path.join(CULANE_ROOT, "driver_161_90frame", "image")
MASK_DIR = os.path.join(CULANE_ROOT, "laneseg_label_w16")
LIST_FILE = os.path.join(CULANE_ROOT, "list", "train_gt.txt")


# ==== Transforms ====
train_transform = A.Compose([
    A.Resize(288, 800),
    A.HorizontalFlip(p=0.5),
    A.RandomBrightnessContrast(p=0.2),
    A.Normalize(mean=(0.485, 0.456, 0.406),
                std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])

# ==== Dataset & Loader ====
dataset = CULaneDataset(IMG_DIR, MASK_DIR, LIST_FILE, transform=train_transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True)

# ==== Model ====
model = SCNN(in_channels=3, num_classes=NUM_CLASSES).to(DEVICE)
criterion = nn.CrossEntropyLoss(ignore_index=255)  # CULane uses 255 as "ignore"
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

# ==== Training Loop ====
for epoch in range(EPOCHS):
    model.train()
    loop = tqdm(loader, desc=f"Epoch [{epoch+1}/{EPOCHS}]", leave=False)

    epoch_loss = 0
    for images, masks in loop:
        images, masks = images.to(DEVICE), masks.to(DEVICE)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()
        loop.set_postfix(loss=loss.item())

    avg_loss = epoch_loss / len(loader)
    print(f"[Epoch {epoch+1}] Loss: {avg_loss:.4f}")

    # Save model checkpoint
    torch.save(model.state_dict(), os.path.join(SAVE_PATH, f"scnn_epoch{epoch+1}.pth"))

    scheduler.step()
