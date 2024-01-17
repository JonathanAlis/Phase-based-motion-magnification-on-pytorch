import numpy as np
import torch
import cv2


import torch
import torch.nn as nn
import torch.nn.functional as F

class SteerablePyramid(nn.Module):
    def __init__(self, levels=3, filters_per_level=6):
        super(SteerablePyramid, self).__init__()
        self.levels = levels
        self.filters_per_level = filters_per_level

        # You can define your own filters based on your requirements
        self.filters = nn.Parameter(torch.randn(levels, filters_per_level))

    def forward(self, x):
        pyramid = []
        for level in range(self.levels):
            print(self.filters[level].shape)
            print(x.shape)
            filtered = F.conv2d(x, self.filters[level].view(1, 1, -1, 1, 1), stride = 1)
            pyramid.append(filtered)

            # Subsample for the next level (you might want to use pooling here)
            x = F.interpolate(x, scale_factor=0.5, mode='nearest')

        return pyramid

class InverseSteerablePyramid(nn.Module):
    def __init__(self, levels=3, filters_per_level=6):
        super(InverseSteerablePyramid, self).__init__()
        self.levels = levels
        self.filters_per_level = filters_per_level

        # You can define your own filters based on your requirements
        self.filters = nn.Parameter(torch.randn(levels, filters_per_level))

    def forward(self, pyramid):
        for level in range(self.levels - 1, -1, -1):
            # Upsample to the original size
            x = F.interpolate(x, scale_factor=2.0, mode='nearest')

            # Apply the inverse filter
            x = F.conv2d(x, self.filters[level].view(1, 1, -1, 1, 1))

            # Add the corresponding level from the pyramid
            x += pyramid[level]

        return x

# Example usage
image = torch.randn(1, 1, 256, 256)  # Example grayscale image

# Forward pass through the steerable pyramid
pyramid_model = SteerablePyramid(levels=3, filters_per_level=6)
pyramid_result = pyramid_model(image)

# Inverse pass to reconstruct the image
inverse_pyramid_model = InverseSteerablePyramid(levels=3, filters_per_level=6)
reconstructed_image = inverse_pyramid_model(pyramid_result)

print(image.shape)
print(pyramid_result.shape)
print(reconstructed_image.shape)


# Requires PyTorch with MKL when setting to 'cpu' 
device = torch.device('cuda:0')

# Initialize Complex Steerbale Pyramid
pyr = SCFpyr_PyTorch(height=5, nbands=4, scale_factor=2, device=device)


cap = cv2.VideoCapture('data/crane_crop.mp4')

import torch
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image

class VideoDataset(Dataset):
    def __init__(self, video_path, transform=None):
        self.video_path = video_path
        self.video_capture = cv2.VideoCapture(video_path)
        self.transform = transform

    def __len__(self):
        return int(self.video_capture.get(cv2.CAP_PROP_FRAME_COUNT))

    def __getitem__(self, idx):
        self.video_capture.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = self.video_capture.read()

        if not ret:
            raise ValueError("Error reading frame from video.")

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(frame)

        if self.transform:
            image = self.transform(image)

        return image

# Set the path to your MP4 video file
video_path = 'path/to/your/video.mp4'

# Define transformations (you can customize these based on your requirements)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Create a dataset
video_dataset = VideoDataset(video_path, transform=transform)

# Create a DataLoader to iterate over frames in batches
batch_size = 32
data_loader = DataLoader(video_dataset, batch_size=batch_size, shuffle=False)

# Iterate through the DataLoader to get batches of frames
for batch in data_loader:
    # The 'batch' variable contains a batch of frames as PyTorch tensors
    # You can perform further processing or analysis on these frames
    print("Batch shape:", batch.shape)
