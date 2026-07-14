"""Training entrypoint for the custom 3D CNN deepfake classifier.

This script trains the model on video clips loaded from the dataset structure:

    data/
        train/real
        train/fake
        val/real
        val/fake

The implementation is intentionally compact and educational while remaining
fully functional and easy to extend.
"""

from __future__ import annotations

import argparse
import os
import random
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from configs.config import Config
from model import SemanticDeepfakeDetector, load_compatible_checkpoint
from video_dataset import VideoDataset


def set_seed(seed: int) -> None:
    """Set all relevant random seeds for reproducible experiments."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def create_dataloader(
    root_dir: str | Path,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    is_train: bool,
    weak_label_dir: str | Path | None = None,
    return_weak_labels: bool = False,
    align_faces: bool = Config.ALIGN_FACES,
) -> DataLoader:
    """Build a dataloader for one dataset split."""

    dataset = VideoDataset(
        root_dir=root_dir,
        clip_length=Config.CLIP_LENGTH,
        frame_size=Config.IMAGE_SIZE,
        is_train=is_train,
        weak_label_dir=weak_label_dir,
        return_weak_labels=return_weak_labels,
        concept_names=Config.CONCEPT_NAMES,
        align_faces=align_faces,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


def compute_accuracy(logits: Tensor, labels: Tensor) -> float:
    """Compute classification accuracy for a batch as a percentage."""

    predictions = torch.argmax(logits, dim=1)
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return 100.0 * correct / max(total, 1)


def train_one_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> Tuple[float, float]:
    """Run one full training epoch and return loss and accuracy."""

    model.train()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    progress = tqdm(dataloader, desc="Train", leave=False)
    for batch in progress:
        if len(batch) == 3:
            clips, labels, weak_labels = batch
            weak_labels = weak_labels.to(device, non_blocking=True)
        else:
            clips, labels = batch
            weak_labels = None

        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(clips, return_dict=True)
        if weak_labels is None:
            loss = nn.functional.cross_entropy(
                outputs["logits"], labels, label_smoothing=Config.LABEL_SMOOTHING
            )
            loss_parts = {"classification_loss": loss.detach()}
        else:
            loss, loss_parts = model.compute_loss(
                outputs=outputs,
                labels=labels,
                weak_labels=weak_labels,
                concept_loss_weight=Config.CONCEPT_LOSS_WEIGHT,
            )
        loss.backward()
        optimizer.step()

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(outputs["logits"], dim=1) == labels).sum().item()
        running_total += batch_size

        postfix = {
            "loss": f"{loss.item():.4f}",
            "ce": f"{loss_parts['classification_loss'].item():.4f}",
        }
        if "concept_loss" in loss_parts:
            postfix["bce"] = f"{loss_parts['concept_loss'].item():.4f}"
        progress.set_postfix(postfix)

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = 100.0 * running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Tuple[float, float]:
    """Evaluate the model on a validation or test loader."""

    model.eval()

    running_loss = 0.0
    running_correct = 0
    running_total = 0

    for clips, labels in tqdm(dataloader, desc="Eval", leave=False):
        clips = clips.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        outputs = model(clips)
        loss = criterion(outputs, labels)

        batch_size = labels.size(0)
        running_loss += loss.item() * batch_size
        running_correct += (torch.argmax(outputs, dim=1) == labels).sum().item()
        running_total += batch_size

    epoch_loss = running_loss / max(running_total, 1)
    epoch_acc = 100.0 * running_correct / max(running_total, 1)
    return epoch_loss, epoch_acc


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: ReduceLROnPlateau | None,
    epoch: int,
    val_accuracy: float,
    checkpoint_path: Path,
) -> None:
    """Persist the best model checkpoint to disk."""

    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    checkpoint: Dict[str, object] = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "val_accuracy": val_accuracy,
        "class_to_idx": {"real": 0, "fake": 1},
        "architecture": "SemanticDeepfakeDetector",
        "concept_vocabulary": Config.CONCEPT_NAMES,
        "concept_loss_weight": Config.CONCEPT_LOSS_WEIGHT,
    }

    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()

    torch.save(checkpoint, checkpoint_path)


def build_parser() -> argparse.ArgumentParser:
    """Create the command-line argument parser."""

    parser = argparse.ArgumentParser(description="Train a custom 3D CNN on video clips.")
    parser.add_argument("--train-dir", type=str, default=Config.TRAIN_DIR)
    parser.add_argument("--val-dir", type=str, default=Config.VAL_DIR)
    parser.add_argument("--epochs", type=int, default=Config.EPOCHS)
    parser.add_argument("--batch-size", type=int, default=Config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=Config.LEARNING_RATE)
    parser.add_argument("--num-workers", type=int, default=Config.NUM_WORKERS)
    parser.add_argument(
        "--weak-label-dir",
        type=str,
        default=None,
        help="Directory containing per-video weak-label JSON/NPY files.",
    )
    parser.add_argument(
        "--pretrained-backbone-path",
        type=str,
        default=None,
        help="Optional old CNN checkpoint used to initialize compatible backbone weights.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=Config.DEVICE,
        help="Device to use for training, e.g. cpu or cuda.",
    )
    parser.add_argument("--checkpoint-path", type=str, default=os.path.join(Config.CHECKPOINT_DIR, Config.CHECKPOINT_NAME))
    parser.add_argument("--no-scheduler", action="store_true", help="Disable the learning-rate scheduler.")
    parser.add_argument(
        "--align-faces",
        action="store_true",
        default=Config.ALIGN_FACES,
        help="Use landmark-based eye alignment before face crop/resize when landmarks are available.",
    )
    return parser


def main() -> None:
    """Train the model and save the best validation checkpoint."""

    parser = build_parser()
    args = parser.parse_args()

    set_seed(Config.SEED)

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot see a GPU. "
            "Install a CUDA-enabled PyTorch build and make sure VS Code is using that interpreter."
        )

    device = torch.device(args.device)

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    train_loader = create_dataloader(
        root_dir=args.train_dir,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        is_train=True,
        weak_label_dir=args.weak_label_dir,
        return_weak_labels=args.weak_label_dir is not None,
        align_faces=args.align_faces,
    )
    val_loader = create_dataloader(
        root_dir=args.val_dir,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        is_train=False,
        weak_label_dir=args.weak_label_dir,
        return_weak_labels=False,
        align_faces=args.align_faces,
    )

    model = SemanticDeepfakeDetector(
        num_classes=Config.NUM_CLASSES,
        concept_vocabulary=Config.CONCEPT_NAMES,
        extra_unsupervised_concepts=Config.EXTRA_UNSUPERVISED_CONCEPTS,
    ).to(device)

    if args.pretrained_backbone_path is not None:
        missing_keys, unexpected_keys = load_compatible_checkpoint(
            model=model,
            checkpoint_path=args.pretrained_backbone_path,
            device=device,
            strict=False,
        )
        print(f"Loaded compatible weights from {args.pretrained_backbone_path}")
        print(f"Missing new-model keys: {len(missing_keys)}")
        print(f"Unexpected checkpoint keys: {len(unexpected_keys)}")

    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=Config.WEIGHT_DECAY)
    scheduler = None
    if not args.no_scheduler:
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=Config.LR_PLATEAU_FACTOR,
            patience=Config.LR_PLATEAU_PATIENCE,
            verbose=True,
        )

    best_val_accuracy = 0.0
    checkpoint_path = Path(args.checkpoint_path)

    print(f"Device: {device}")
    print(f"Training samples: {len(train_loader.dataset)}")
    print(f"Validation samples: {len(val_loader.dataset)}")

    criterion = nn.CrossEntropyLoss()

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            dataloader=train_loader,
            optimizer=optimizer,
            device=device,
        )
        val_loss, val_accuracy = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
        )

        if scheduler is not None:
            scheduler.step(val_loss)

        print(
            f"Epoch {epoch:02d}/{args.epochs:02d} | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_accuracy:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_accuracy:.2f}%"
        )

        if val_accuracy > best_val_accuracy:
            best_val_accuracy = val_accuracy
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                val_accuracy=val_accuracy,
                checkpoint_path=checkpoint_path,
            )
            print(f"Saved best model to {checkpoint_path}")

    print(f"Best validation accuracy: {best_val_accuracy:.2f}%")


if __name__ == "__main__":
    main()
