import torch
import torch.nn as nn


class CNN3D(nn.Module):
    """
    Lightweight 3D CNN for binary video classification.

    Input Shape:
        (B, 3, 16, 112, 112)

    Output:
        (B, 2)
    """

    def __init__(self, num_classes=2):
        super(CNN3D, self).__init__()

        # -------------------------
        # Block 1
        # -------------------------
        self.conv1 = nn.Conv3d(
            in_channels=3,
            out_channels=32,
            kernel_size=3,
            padding=1
        )

        self.bn1 = nn.BatchNorm3d(32)

        self.pool1 = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # -------------------------
        # Block 2
        # -------------------------
        self.conv2 = nn.Conv3d(
            32,
            64,
            kernel_size=3,
            padding=1
        )

        self.bn2 = nn.BatchNorm3d(64)

        self.pool2 = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # -------------------------
        # Block 3
        # -------------------------
        self.conv3 = nn.Conv3d(
            64,
            128,
            kernel_size=3,
            padding=1
        )

        self.bn3 = nn.BatchNorm3d(128)

        self.pool3 = nn.MaxPool3d(
            kernel_size=2,
            stride=2
        )

        # -------------------------
        # Global Average Pool
        # -------------------------
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))

        # -------------------------
        # Classifier
        # -------------------------
        self.classifier = nn.Sequential(

            nn.Flatten(),

            nn.Linear(128, 64),

            nn.ReLU(inplace=True),

            nn.Dropout(0.3),

            nn.Linear(64, num_classes)
        )

    def forward(self, x):

        # Block 1
        x = self.pool1(
            torch.relu(
                self.bn1(
                    self.conv1(x)
                )
            )
        )

        # Block 2
        x = self.pool2(
            torch.relu(
                self.bn2(
                    self.conv2(x)
                )
            )
        )

        # Block 3
        x = self.pool3(
            torch.relu(
                self.bn3(
                    self.conv3(x)
                )
            )
        )

        # Global Average Pool
        x = self.avgpool(x)

        # Classifier
        x = self.classifier(x)

        return x