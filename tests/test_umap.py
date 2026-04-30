from thesis_assyrian_relief.evaluation.retrieval import extract_embeddings
from thesis_assyrian_relief.utils.plotting import (
    build_umap_dataframe,
    plot_umap_matplotlib,
    plot_umap_plotly,
)
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe

import warnings
warnings.filterwarnings("ignore", message="xFormers is not available")

# assumes you already have:
# train_relief_emb_df
# test_relief_emb_df

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

test_ds = ReliefStyleDataset(
    csv_path="data/splits/image_level_dataset.csv",
    split="test",
    class_to_idx=class_to_idx,
    transform=transform,
    image_root="/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset",
    check_paths=False,
)

train_loader = DataLoader(train_ds, batch_size=4, shuffle=False)
test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DinoStyleProbe(
    num_classes=len(class_to_idx),
    emb_dim=256,
).to(device)

train_relief_emb_df = extract_embeddings(model, train_loader, device)
test_relief_emb_df = extract_embeddings(model, test_loader, device)

viz_df = build_umap_dataframe(
    relief_emb_dfs=[train_relief_emb_df, test_relief_emb_df],
    split_names=["train", "test"],
)

print(viz_df.head())

fig = plot_umap_plotly(
    viz_df,
    highlight_relief_ids=["BM 124773", "BM 124788"],
)
# fig.show()

output_html = "/mnt/c/Users/chend/Desktop/My_Files/Thesis/outputs/umap_plot.html"
fig.write_html(output_html, include_plotlyjs="cdn")
print(f"Saved interactive plot to: {output_html}")