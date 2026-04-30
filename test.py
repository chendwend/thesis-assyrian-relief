from thesis_assyrian_relief.utils.data import (
    build_class_to_idx,
    build_dataloader,
    build_class_weights,
)

class_to_idx = build_class_to_idx("data/splits/image_level_dataset.csv")
print(class_to_idx)

train_ds, train_loader = build_dataloader(
    csv_path="data/splits/image_level_dataset.csv",
    split="train",
    class_to_idx=class_to_idx,
    image_root="/mnt/c/Users/chend/Desktop/My_Files/Thesis/Dataset/dataset",
    batch_size=4,
    num_workers=0,
    shuffle=True,
    train=True,
    check_paths=False,
)

print(len(train_ds), len(train_loader))

weights = build_class_weights(train_ds, class_to_idx)
print(weights)