from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class DinoStyleProbe(nn.Module):
    """
    DINOv2 frozen-backbone style model.

    Design:
        image -> frozen DINOv2 backbone -> trainable embedding head -> normalized embedding -> classifier

    Purpose:
        - classification by monarch/style class
        - extraction of a meaningful embedding space for centroiding, retrieval, and visualization
    """

    def __init__(
        self,
        num_classes: int,
        emb_dim: int = 256,
        model_name: str = "dinov2_vits14",
        pretrained: bool = True,
        freeze_backbone: bool = True,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()

        # Official Facebook Research DINOv2 hub model
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2",
            model_name,
            pretrained=pretrained,
        )

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

        feat_dim = self._infer_feature_dim()

        self.embedding_head = nn.Sequential(
            nn.Linear(feat_dim, emb_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(emb_dim, emb_dim),
        )

        self.classifier = nn.Linear(emb_dim, num_classes)

    def _infer_feature_dim(self) -> int:
        """
        Infer backbone output dimension using a dummy forward pass.
        Assumes 224x224 RGB input, which is fine for DINOv2 probe setup.
        """
        was_training = self.backbone.training
        self.backbone.eval()

        with torch.no_grad():
            dummy = torch.zeros(1, 3, 224, 224)
            feat = self.backbone(dummy)

        if was_training:
            self.backbone.train()

        if feat.ndim != 2:
            raise ValueError(
                f"Expected DINOv2 backbone output shape [B, D], got {tuple(feat.shape)}"
            )

        return int(feat.shape[1])

    def extract_backbone_features(self, x: torch.Tensor) -> torch.Tensor:
        """
        Return backbone features before the trainable embedding head.
        """
        if any(p.requires_grad for p in self.backbone.parameters()):
            return self.backbone(x)

        with torch.no_grad():
            return self.backbone(x)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
            logits: [B, num_classes]
            emb:    [B, emb_dim] normalized embedding
        """
        feats = self.extract_backbone_features(x)
        emb = self.embedding_head(feats)
        emb = F.normalize(emb, p=2, dim=1)
        logits = self.classifier(emb)
        return logits, emb