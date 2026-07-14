"""Inference-only helpers for inspecting semantic attention heads."""

from __future__ import annotations

from typing import List, Mapping

import torch
from torch import Tensor


def _normalized_entropy(probabilities: Tensor) -> Tensor:
    token_count = probabilities.shape[-1]
    entropy = -(probabilities * torch.log(probabilities.clamp_min(1e-8))).sum(dim=-1)
    if token_count <= 1:
        return torch.zeros_like(entropy)
    return entropy / torch.log(torch.tensor(float(token_count), device=probabilities.device))


@torch.no_grad()
def summarize_head_outputs(outputs: Mapping[str, Tensor]) -> List[Mapping[str, float | int]]:
    """Summarize head energy and attention concentration from model outputs.

    The returned values are descriptive only. They do not assign fixed meanings
    such as boundary, eyes, mouth, or motion; use them to compare head behavior
    on a clip or across a dataset without changing training.
    """

    head_feature_maps = outputs["head_feature_maps"].detach()
    attention_probs = outputs["attention_probs"].detach()
    head_weights = outputs["head_weights"].detach()
    head_vectors = head_feature_maps.mean(dim=(3, 4, 5))

    summaries: List[Mapping[str, float | int]] = []
    for head_index in range(head_feature_maps.shape[1]):
        features = head_feature_maps[:, head_index]
        attention = attention_probs[:, head_index]
        entropy = _normalized_entropy(attention)
        summaries.append(
            {
                "head_index": head_index,
                "fusion_weight": float(head_weights[head_index].cpu().item()),
                "feature_mean_abs": float(features.abs().mean().cpu().item()),
                "feature_l2": float(torch.linalg.vector_norm(head_vectors[:, head_index], dim=1).mean().cpu().item()),
                "attention_peak_mean": float(attention.max(dim=-1).values.mean().cpu().item()),
                "attention_entropy": float(entropy.mean().cpu().item()),
                "attention_concentration": float((1.0 - entropy).mean().cpu().item()),
            }
        )
    return summaries
