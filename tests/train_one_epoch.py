import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.training.engine import train_one_epoch, evaluate

local_root ="/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset"
selected_root = local_root

class_to_idx = {
    "Ashurbanipal": 0,
    "Ashurnasirpal II": 1,
    "OTHER": 2,
    "Sargon II": 3,
}

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

train_ds = ReliefStyleDataset(
    csv_path="data/splits/image_level_dataset.csv",
    split="train",
    class_to_idx=class_to_idx,
    transform=transform,
    image_root=selected_root,
    filename_sep="-",
    check_paths=False,
)

val_ds = ReliefStyleDataset(
    csv_path="data/splits/image_level_dataset.csv",
    split="val",
    class_to_idx=class_to_idx,
    transform=transform,
    image_root=selected_root,
    filename_sep="-",
    check_paths=False,
)

train_loader = DataLoader(train_ds, batch_size=2, shuffle=True)
val_loader = DataLoader(val_ds, batch_size=2, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DinoStyleProbe(num_classes=4, emb_dim=256).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=1e-3,
)

train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
val_metrics = evaluate(model, val_loader, criterion, device)

print(train_metrics)
print(val_metrics)