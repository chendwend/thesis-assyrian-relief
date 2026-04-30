import torch
from thesis_assyrian_relief.models.dinov2_probe import DinoStyleProbe

model = DinoStyleProbe(
    num_classes=4, # TODO: make this configurable
    emb_dim=256,
    model_name="dinov2_vits14",
)

x = torch.randn(2, 3, 224, 224)
logits, emb = model(x)

print("logits shape:", logits.shape)
print("embedding shape:", emb.shape)
print("embedding norms:", emb.norm(dim=1))