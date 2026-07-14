"""Custom lightweight 3D CNN for binary deepfake video classification.

This module keeps the original backbone available for compatibility while also
providing a semantic attention model that consumes per-video weak labels.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Sequence, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F


DEFAULT_CONCEPT_VOCABULARY: Tuple[str, str] = (
    "boundary_inconsistency",
    "eye_blink_irregularity",
)


class Residual3DBlock(nn.Module):
    """A lightweight residual block for spatiotemporal feature refinement."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(channels)

    def forward(self, x: Tensor) -> Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.relu(out + identity)
        return out


class Custom3DCNN(nn.Module):
    """A compact 3D CNN for classifying video clips as real or fake.

    The network expects input tensors in ``(B, 3, T, H, W)`` format, for
    example ``(B, 3, 16, 112, 112)``.
    """

    def __init__(self, num_classes: int = 2, dropout: float = 0.4) -> None:
        super().__init__()

        self.conv1 = nn.Conv3d(3, 24, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(24)
        self.relu1 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.res2 = Residual3DBlock(24)
        self.conv2 = nn.Conv3d(24, 48, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(48)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.res3 = Residual3DBlock(48)
        self.conv3 = nn.Conv3d(48, 96, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm3d(96)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool3 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.final_conv = nn.Conv3d(96, 128, kernel_size=3, padding=1, bias=False)
        self.final_bn = nn.BatchNorm3d(128)
        self.final_relu = nn.ReLU(inplace=True)

        self.global_pool = nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(128, num_classes),
        )

        self._init_weights()

    def _init_weights(self) -> None:
        """Initialize weights with Kaiming initialization for stable training."""

        for module in self.modules():
            if isinstance(module, nn.Conv3d):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.BatchNorm3d):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

    def forward_feature_map(self, x: Tensor) -> Tensor:
        """Run the convolutional backbone and keep the 3D feature map."""

        x = self.pool1(self.relu1(self.bn1(self.conv1(x))))
        x = self.res2(x)
        x = self.pool2(self.relu2(self.bn2(self.conv2(x))))
        x = self.res3(x)
        x = self.pool3(self.relu3(self.bn3(self.conv3(x))))
        x = self.final_relu(self.final_bn(self.final_conv(x)))
        return x

    def forward_features(self, x: Tensor) -> Tensor:
        """Run the convolutional feature extractor and return pooled features."""

        x = self.forward_feature_map(x)
        x = self.global_pool(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        """Compute class logits for a batch of video clips."""

        features = self.forward_features(x)
        logits = self.classifier(features)
        return logits


class CNN3D(Custom3DCNN):
    """Backward-compatible alias used by some scripts in the repository."""


class SpatiotemporalMultiHeadAttention(nn.Module):
    """Multi-head attention over flattened ``(T, H, W)`` CNN feature tokens."""

    def __init__(self, channels: int, num_heads: int = 4) -> None:
        super().__init__()
        if channels % num_heads != 0:
            raise ValueError("channels must be divisible by num_heads")

        self.channels = channels
        self.num_heads = num_heads
        self.head_dim = channels // num_heads
        self.scale = self.head_dim ** -0.5
        self.qkv = nn.Linear(channels, channels * 3)

    def forward(self, feature_map: Tensor) -> Tuple[Tensor, Tensor]:
        """Return per-head feature maps and attention probabilities."""

        batch_size, channels, depth, height, width = feature_map.shape
        tokens = feature_map.flatten(2).transpose(1, 2)

        qkv = self.qkv(tokens)
        qkv = qkv.reshape(batch_size, -1, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        query, key, value = qkv.unbind(dim=0)

        attention_logits = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        attention_probs = torch.softmax(attention_logits, dim=-1)
        attended = torch.matmul(attention_probs, value)

        head_feature_maps = attended.transpose(2, 3).reshape(
            batch_size,
            self.num_heads,
            self.head_dim,
            depth,
            height,
            width,
        )
        return head_feature_maps, attention_probs


class HeadWeightedFusion(nn.Module):
    """Learned softmax weighting over attention heads."""

    def __init__(self, num_heads: int = 4) -> None:
        super().__init__()
        self.logits = nn.Parameter(torch.zeros(num_heads))

    def forward(self, head_vectors: Tensor) -> Tuple[Tensor, Tensor]:
        """Fuse head vectors with learned normalized scalar weights."""

        weights = torch.softmax(self.logits, dim=0)
        fused = (head_vectors * weights.view(1, -1, 1)).sum(dim=1)
        return fused, weights


class SemanticDeepfakeDetector(Custom3DCNN):
    """3D CNN with interpretable semantic attention and concept bottleneck."""

    def __init__(
        self,
        num_classes: int = 2,
        dropout: float = 0.4,
        concept_vocabulary: Sequence[str] = DEFAULT_CONCEPT_VOCABULARY,
        extra_unsupervised_concepts: Sequence[str] | None = None,
        num_attention_heads: int = 4,
    ) -> None:
        super().__init__(num_classes=num_classes, dropout=dropout)

        self.concept_vocabulary = tuple(concept_vocabulary)
        self.extra_unsupervised_concepts = tuple(extra_unsupervised_concepts or ())
        self.num_concepts = len(self.concept_vocabulary)
        self.num_attention_heads = num_attention_heads

        self.semantic_attention = SpatiotemporalMultiHeadAttention(channels=128, num_heads=num_attention_heads)
        self.head_fusion = HeadWeightedFusion(num_heads=num_attention_heads)
        self.concept_projection = nn.Sequential(
            nn.Linear(128 // num_attention_heads, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, self.num_concepts),
        )
        self.concept_embedding = nn.Sequential(
            nn.Linear(self.num_concepts, self.num_concepts),
            nn.LayerNorm(self.num_concepts),
        )
        self.semantic_classifier = nn.Linear(self.num_concepts, num_classes)
        # A direct visual branch prevents classification from being forced
        # through only two weakly/unsupervised concept values. The semantic
        # branch remains available as an auxiliary, interpretable signal.
        head_width = 128 // num_attention_heads
        self.visual_classifier = nn.Sequential(
            nn.Linear(head_width, 64),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(64, num_classes),
        )
        self.semantic_logit_gate = nn.Parameter(torch.tensor(-2.0))

        self._init_semantic_weights()

    def _init_semantic_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.kaiming_normal_(module.weight, mode="fan_out", nonlinearity="relu")
                if module.bias is not None:
                    nn.init.zeros_(module.bias)

    def forward(self, x: Tensor, return_dict: bool = False) -> Tensor | Dict[str, Tensor]:
        """Compute class logits, optionally with semantic intermediate tensors."""

        backbone_feature_map = self.forward_feature_map(x)
        head_feature_maps, attention_probs = self.semantic_attention(backbone_feature_map)
        head_vectors = head_feature_maps.mean(dim=(3, 4, 5))
        fused_vector, head_weights = self.head_fusion(head_vectors)

        concept_logits = self.concept_projection(fused_vector)
        concept_scores = torch.sigmoid(concept_logits)
        concept_embedding = self.concept_embedding(concept_scores)
        semantic_logits = self.semantic_classifier(concept_embedding)
        visual_logits = self.visual_classifier(fused_vector)
        semantic_gate = torch.sigmoid(self.semantic_logit_gate)
        logits = visual_logits + semantic_gate * semantic_logits

        if not return_dict:
            return logits

        return {
            "logits": logits,
            "concept_logits": concept_logits,
            "concept_scores": concept_scores,
            "concept_embedding": concept_embedding,
            "visual_logits": visual_logits,
            "semantic_logits": semantic_logits,
            "semantic_gate": semantic_gate,
            "head_feature_maps": head_feature_maps,
            "attention_probs": attention_probs,
            "head_weights": head_weights,
            "fused_vector": fused_vector,
            "backbone_feature_map": backbone_feature_map,
        }

    def compute_loss(
        self,
        outputs: Dict[str, Tensor],
        labels: Tensor,
        weak_labels: Tensor,
        concept_loss_weight: float = 0.3,
    ) -> Tuple[Tensor, Dict[str, Tensor]]:
        """Compute CE + weighted mean BCE for supervised concepts."""

        classification_loss = F.cross_entropy(
            outputs["logits"], labels, label_smoothing=getattr(Config, "LABEL_SMOOTHING", 0.0)
        )
        concept_loss = F.binary_cross_entropy(outputs["concept_scores"], weak_labels.float())
        total_loss = classification_loss + concept_loss_weight * concept_loss
        return total_loss, {
            "classification_loss": classification_loss.detach(),
            "concept_loss": concept_loss.detach(),
            "total_loss": total_loss.detach(),
        }


def build_model(num_classes: int = 2) -> Custom3DCNN:
    """Factory helper for scripts that prefer a function-based constructor."""

    return Custom3DCNN(num_classes=num_classes)


def build_semantic_model(
    num_classes: int = 2,
    concept_vocabulary: Sequence[str] = DEFAULT_CONCEPT_VOCABULARY,
) -> SemanticDeepfakeDetector:
    """Factory helper for the semantic attention model."""

    return SemanticDeepfakeDetector(num_classes=num_classes, concept_vocabulary=concept_vocabulary)


def extract_state_dict(checkpoint: Any) -> Dict[str, Tensor]:
    """Extract model weights from common PyTorch checkpoint formats."""

    if isinstance(checkpoint, nn.Module):
        return checkpoint.state_dict()
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        return checkpoint["model_state_dict"]
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        return checkpoint["state_dict"]
    if isinstance(checkpoint, dict):
        return checkpoint
    raise TypeError(f"Unsupported checkpoint type: {type(checkpoint)!r}")


def _strip_module_prefix(state_dict: Dict[str, Tensor]) -> Dict[str, Tensor]:
    if not any(key.startswith("module.") for key in state_dict):
        return state_dict
    return {key.removeprefix("module."): value for key, value in state_dict.items()}


def load_compatible_checkpoint(
    model: nn.Module,
    checkpoint_path: str | Path,
    device: torch.device,
    strict: bool = False,
) -> Tuple[Tuple[str, ...], Tuple[str, ...]]:
    """Load a checkpoint and report key mismatches.

    When ``strict=False``, only parameters that exist in the target model and
    have matching shapes are loaded. This keeps old CNN checkpoints usable as a
    backbone warm-start for the semantic model.
    """

    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        # Compatibility with PyTorch versions released before ``weights_only``.
        checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = _strip_module_prefix(extract_state_dict(checkpoint))

    if not strict:
        model_state = model.state_dict()
        state_dict = {
            key: value
            for key, value in state_dict.items()
            if key in model_state and tuple(value.shape) == tuple(model_state[key].shape)
        }

    load_result = model.load_state_dict(state_dict, strict=strict)
    return tuple(load_result.missing_keys), tuple(load_result.unexpected_keys)
