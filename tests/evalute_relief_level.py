import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms

from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.evaluation.relief_level import evaluate_relief_level

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

val_ds = ReliefStyleDataset(
    csv_path="data/splits/image_level_dataset.csv",
    split="val",
    class_to_idx=class_to_idx,
    transform=transform,
    image_root= "/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset",
    check_paths=False,
)

val_loader = DataLoader(
    val_ds,
    batch_size=4,
    shuffle=False,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DinoStyleProbe(
    num_classes=len(class_to_idx),
    emb_dim=256,
).to(device)

metrics, relief_df = evaluate_relief_level(model, val_loader, device)

print(metrics)
print(relief_df.head())