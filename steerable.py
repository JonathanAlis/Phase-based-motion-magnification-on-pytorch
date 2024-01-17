import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def gaussian(x, sigma):
    return np.exp(-(x ** 2) / (2 * sigma ** 2))

def steerable_filters(angles, scale, spatial_extent, orientation_extent):
    filters = []
    for angle in angles:
        # Generate a Gabor filter
        x, y = np.meshgrid(np.linspace(-spatial_extent, spatial_extent, scale),
                           np.linspace(-spatial_extent, spatial_extent, scale))

        x_rot = x * np.cos(angle) + y * np.sin(angle)
        y_rot = -x * np.sin(angle) + y * np.cos(angle)

        gabor = np.exp(-(x_rot ** 2 + y_rot ** 2) / (2 * orientation_extent ** 2)) * np.cos(2 * np.pi * x_rot / scale)

        # Normalize the filter
        gabor /= np.sum(np.abs(gabor))

        filters.append(gabor)

    return filters

class ComplexSteerablePyramid(nn.Module):
    def __init__(self, levels=3, filters_per_level=6):
        super(ComplexSteerablePyramid, self).__init__()
        self.levels = levels
        self.filters_per_level = filters_per_level

        # Steerable filters
        self.filters = nn.Parameter(torch.randn(filters_per_level, 1, 8, 8))

    def forward(self, x):
        pyramid_real = []
        pyramid_imag = []

        for level in range(self.levels):
            # Apply steerable filter
            filtered_real = F.conv2d(x, self.filters[level].unsqueeze(0), stride=1, padding=0)
            filtered_imag = F.conv2d(x, self.filters[level].unsqueeze(0), stride=1, padding=0)

            pyramid_real.append(filtered_real)
            pyramid_imag.append(filtered_imag)

            # Subsample for the next level (you might want to use pooling here)
            x = F.interpolate(x, scale_factor=0.5, mode='nearest')

        return pyramid_real, pyramid_imag

class InverseComplexSteerablePyramid(nn.Module):
    def __init__(self, levels=3, filters_per_level=6):
        super(InverseComplexSteerablePyramid, self).__init__()
        self.levels = levels
        self.filters_per_level = filters_per_level

        # Use the same steerable filters for the inverse pyramid
        self.filters = nn.Parameter(torch.randn(filters_per_level, 1, 8, 8))

    def forward(self, pyramid_real, pyramid_imag):
        # Initialize the reconstructed image with the finest scale
        reconstructed_image = pyramid_real[-1] + 1j * pyramid_imag[-1]

        for level in reversed(range(self.levels - 1)):
            # Upsample the reconstructed image
            reconstructed_image = F.interpolate(reconstructed_image.real.unsqueeze(1), scale_factor=2, mode='nearest') + \
                                  1j * F.interpolate(reconstructed_image.imag.unsqueeze(1), scale_factor=2, mode='nearest')
            print(reconstructed_image.real.shape)
            # Reshape the reconstructed image to have the shape (minibatch, 1, height, width)
            # Apply the inverse steerable filter using transposed convolution
            filtered_real = F.conv_transpose2d(reconstructed_image.real, self.filters[level].unsqueeze(0), stride=1, padding=0)
            filtered_imag = F.conv_transpose2d(reconstructed_image.imag, self.filters[level].unsqueeze(0), stride=1, padding=0)

            # Update the reconstructed image
            reconstructed_image = filtered_real + 1j * filtered_imag

            # Add the corresponding level from the inverse pyramid
            reconstructed_image += pyramid_real[level] + 1j * pyramid_imag[level]

        return reconstructed_image


# Example usage
image = torch.randn(1, 1, 256, 256)  # Example grayscale image
steerable_pyramid = ComplexSteerablePyramid()
pyramid_real, pyramid_imag = steerable_pyramid(image)
for pyr in pyramid_imag:
    print(pyr.shape)
for pyr in pyramid_real:
    print(pyr.shape)

# Example usage
inverse_pyramid_model = InverseComplexSteerablePyramid()
reconstructed_image = inverse_pyramid_model(pyramid_real, pyramid_imag)

print(reconstructed_image.shape)