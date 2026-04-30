import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe
from thesis_assyrian_relief.evaluation.retrieval import (
    extract_embeddings,
    aggregate_relief_embeddings,
    build_class_centroids,
    predict_by_nearest_centroid,
    compute_retrieval_metrics,
)

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
    image_root="/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset",
    check_paths=False,
)

val_ds = ReliefStyleDataset(
    csv_path="data/splits/image_level_dataset.csv",
    split="val",
    class_to_idx=class_to_idx,
    transform=transform,
    image_root="/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset",
    check_paths=False,
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
val_loader = DataLoader(val_ds, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DinoStyleProbe(
    num_classes=len(class_to_idx),
    emb_dim=256,
).to(device)

train_emb_df = extract_embeddings(model, train_loader, device)
val_emb_df = extract_embeddings(model, val_loader, device)

train_relief_emb_df = aggregate_relief_embeddings(train_emb_df)
val_relief_emb_df = aggregate_relief_embeddings(val_emb_df)

centroids = build_class_centroids(train_relief_emb_df)
y_true, y_pred = predict_by_nearest_centroid(val_relief_emb_df, centroids)

print("n_train_reliefs:", len(train_relief_emb_df))
print("n_val_reliefs:", len(val_relief_emb_df))
print("first true/pred:", y_true[:3], y_pred[:3])

metrics, results_df = compute_retrieval_metrics(
    query_df=val_relief_emb_df,
    gallery_df=train_relief_emb_df,
    ks=(1, 3),
)

print(metrics)
print(results_df.head())
