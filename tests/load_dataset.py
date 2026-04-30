from torchvision import transforms
from thesis_assyrian_relief.datasets.relief_style_dataset import ReliefStyleDataset

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

class_to_idx = {
    "Ashurbanipal": 0,
    "Ashurnasirpal II": 1,
    "OTHER": 2,
    "Sargon II": 3,
}


remote_root ="/content/dataset"
local_root ="/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset"

ds = ReliefStyleDataset(
    csv_path="data/splits/image_level_dataset.csv",
    split="train",
    class_to_idx=class_to_idx,
    transform=transform,
    image_root=local_root,
    filename_sep="-",  
)

print(len(ds))
sample = ds[0]
print(sample["image"].shape)
print(sample["relief_id"], sample["view_index"], sample["suffix"], sample["image_path"])