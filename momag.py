import torch
import torchvision
from torchvision.io import read_video
from torch.utils.data import DataLoader, Dataset
import os

# Define a custom dataset to load video frames
class VideoDataset(Dataset):
    def __init__(self, video_path):
        self.video_path = video_path
        self.video_frames, _, _ = read_video(video_path)
        self.video_frames = self.video_frames.permute(0, 3, 1, 2)  # Permute dimensions for (batch, channels, height, width)

    def __len__(self):
        return len(self.video_frames)

    def __getitem__(self, idx):
        return self.video_frames[idx]

# Path to your video file
video_path = 'data/baby.mp4'

# Create a VideoDataset instance
video_dataset = VideoDataset(video_path)

# Create a DataLoader for batching
batch_size = 4
video_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=True)

# Iterate over batches
for batch in video_loader:
    # Process each batch as needed
    print("Batch shape:", batch.shape)
    # Your processing code here

    