import os

import torch
import torch.nn as nn
from tqdm import tqdm

from configs.config import Config


class Trainer:

    def __init__(
        self,
        model,
        train_loader,
        val_loader,
        optimizer,
        scheduler=None
    ):

        self.model = model.to(Config.DEVICE)

        self.train_loader = train_loader
        self.val_loader = val_loader

        self.optimizer = optimizer
        self.scheduler = scheduler

        self.criterion = nn.CrossEntropyLoss()

        self.best_accuracy = 0.0

    # --------------------------------------------------

    def train(self):

        for epoch in range(Config.EPOCHS):

            print(f"\nEpoch {epoch+1}/{Config.EPOCHS}")

            train_loss, train_acc = self.train_one_epoch()

            val_loss, val_acc = self.validate()

            print(f"Train Loss : {train_loss:.4f}")
            print(f"Train Acc  : {train_acc:.2f}%")

            print(f"Val Loss   : {val_loss:.4f}")
            print(f"Val Acc    : {val_acc:.2f}%")

            if self.scheduler is not None:
                self.scheduler.step()

            if val_acc > self.best_accuracy:

                self.best_accuracy = val_acc

                self.save_checkpoint()

                print("Best model saved.")

        print("\nTraining Complete.")
        print(f"Best Validation Accuracy : {self.best_accuracy:.2f}%")

    # --------------------------------------------------

    def train_one_epoch(self):

        self.model.train()

        running_loss = 0.0

        correct = 0
        total = 0

        loop = tqdm(self.train_loader)

        for clips, labels in loop:

            clips = clips.to(Config.DEVICE)

            labels = labels.to(Config.DEVICE)

            self.optimizer.zero_grad()

            outputs = self.model(clips)

            loss = self.criterion(outputs, labels)

            loss.backward()

            self.optimizer.step()

            running_loss += loss.item()

            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)

            correct += (predicted == labels).sum().item()

            loop.set_postfix(loss=loss.item())

        epoch_loss = running_loss / len(self.train_loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    # --------------------------------------------------

    def validate(self):

        self.model.eval()

        running_loss = 0.0

        correct = 0
        total = 0

        with torch.no_grad():

            for clips, labels in self.val_loader:

                clips = clips.to(Config.DEVICE)

                labels = labels.to(Config.DEVICE)

                outputs = self.model(clips)

                loss = self.criterion(outputs, labels)

                running_loss += loss.item()

                _, predicted = torch.max(outputs, 1)

                total += labels.size(0)

                correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(self.val_loader)

        epoch_acc = 100 * correct / total

        return epoch_loss, epoch_acc

    # --------------------------------------------------

    def save_checkpoint(self):

        os.makedirs(
            Config.CHECKPOINT_DIR,
            exist_ok=True
        )

        path = os.path.join(
            Config.CHECKPOINT_DIR,
            Config.CHECKPOINT_NAME
        )

        torch.save(
            self.model.state_dict(),
            path
        )