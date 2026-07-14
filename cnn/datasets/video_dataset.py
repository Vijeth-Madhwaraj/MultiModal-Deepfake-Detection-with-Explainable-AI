import os
import random
import cv2
import numpy as np
import torch

from torch.utils.data import Dataset

from configs.config import Config


class VideoDataset(Dataset):
    """
    Dataset for loading videos for 3D CNN.
    Folder structure:

    train/
        real/
        fake/

    val/
        real/
        fake/
    """

    def __init__(self, root_dir):

        self.root_dir = root_dir

        self.video_paths = []
        self.labels = []

        self.classes = {
            "real": 0,
            "fake": 1
        }

        self._load_dataset()

    def _load_dataset(self):

        for class_name, label in self.classes.items():

            class_dir = os.path.join(self.root_dir, class_name)

            if not os.path.exists(class_dir):
                continue

            for file in os.listdir(class_dir):

                if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):

                    self.video_paths.append(
                        os.path.join(class_dir, file)
                    )

                    self.labels.append(label)

    def __len__(self):

        return len(self.video_paths)

    def _read_video(self, video_path):

        cap = cv2.VideoCapture(video_path)

        frames = []

        while True:

            success, frame = cap.read()

            if not success:
                break

            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            frame = cv2.resize(
                frame,
                (Config.IMAGE_SIZE, Config.IMAGE_SIZE)
            )

            frames.append(frame)

        cap.release()

        return frames

    def _sample_clip(self, frames):

        clip_length = Config.CLIP_LENGTH
        stride = Config.FRAME_STRIDE

        required = clip_length * stride

        if len(frames) >= required:

            max_start = len(frames) - required

            start = random.randint(0, max_start)

            clip = frames[start:start+required:stride]

        else:

            clip = frames.copy()

            while len(clip) < clip_length:

                clip.append(clip[-1])

            clip = clip[:clip_length]

        return clip

    def __getitem__(self, index):

        video_path = self.video_paths[index]

        label = self.labels[index]

        frames = self._read_video(video_path)

        clip = self._sample_clip(frames)

        clip = np.array(clip).astype(np.float32)

        clip /= 255.0

        clip = np.transpose(
            clip,
            (3, 0, 1, 2)
        )

        clip = torch.tensor(
            clip,
            dtype=torch.float32
        )

        label = torch.tensor(
            label,
            dtype=torch.long
        )

        return clip, label