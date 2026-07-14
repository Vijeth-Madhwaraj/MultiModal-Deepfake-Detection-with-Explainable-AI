import os
import torch


class Config:
    """
    Global configuration for the 3D CNN Deepfake Detection project.
    """

    # ==========================================================
    # Paths
    # ==========================================================

    # configs/ now lives inside cnn/, while datasets and generated artifacts
    # remain at the repository root.
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    DATA_DIR = os.path.join(ROOT_DIR, "data")

    TRAIN_DIR = os.path.join(DATA_DIR, "train")
    VAL_DIR = os.path.join(DATA_DIR, "val")
    TEST_DIR = os.path.join(DATA_DIR, "test")

    CHECKPOINT_DIR = os.path.join(ROOT_DIR, "checkpoints")
    OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs")
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    # ==========================================================
    # Device
    # ==========================================================

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    # ==========================================================
    # Dataset
    # ==========================================================

    NUM_CLASSES = 2

    CLASS_NAMES = [
        "real",
        "fake"
    ]

    IMAGE_SIZE = 112

    CLIP_LENGTH = 16

    FRAME_STRIDE = 2

    ALIGN_FACES = False

    # ==========================================================
    # Training
    # ==========================================================

    BATCH_SIZE = 16

    NUM_WORKERS = 4

    EPOCHS = 20

    LEARNING_RATE = 1e-4

    WEIGHT_DECAY = 1e-5

    # Weight applied to the weak-label concept BCE loss. The total training
    # objective is classification_loss + CONCEPT_LOSS_WEIGHT * concept_loss.
    CONCEPT_LOSS_WEIGHT = 0.3

    LABEL_SMOOTHING = 0.05

    # ==========================================================
    # Optimizer
    # ==========================================================

    OPTIMIZER = "Adam"

    # ==========================================================
    # Scheduler
    # ==========================================================

    USE_SCHEDULER = True

    LR_PLATEAU_PATIENCE = 4

    LR_PLATEAU_FACTOR = 0.5

    # ==========================================================
    # Checkpoints
    # ==========================================================

    SAVE_BEST_ONLY = True

    CHECKPOINT_NAME = "best_model.pth"

    # ==========================================================
    # Random Seed
    # ==========================================================

    SEED = 42

    # ==========================================================
    # GradCAM
    # ==========================================================

    TARGET_LAYER = "final_conv"

    HEATMAP_ALPHA = 0.4

    HEAD_NAMES = [
        "head_1",
        "head_2",
        "head_3",
        "head_4"
    ]

    CONCEPT_NAMES = [
        "boundary_inconsistency",
        "eye_blink_irregularity"
    ]

    EXTRA_UNSUPERVISED_CONCEPTS = []
