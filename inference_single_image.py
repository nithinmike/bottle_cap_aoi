from pathlib import Path
import numpy as np
import os, shutil
import matplotlib.pyplot as plt

from PIL import Image

from tqdm.auto import tqdm

import torch
import torchvision
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
import torch.optim as optim

from torchvision.models import resnet50, ResNet50_Weights

import torch.nn as nn

class FeatCAE(nn.Module):
    """Autoencoder."""

    def __init__(self, in_channels=1000, latent_dim=50, is_bn=True):
        super(FeatCAE, self).__init__()

        layers = []
        layers += [nn.Conv2d(in_channels, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, latent_dim, kernel_size=1, stride=1, padding=0)]

        self.encoder = nn.Sequential(*layers)

        layers = []
        layers += [nn.Conv2d(latent_dim, 2 * latent_dim, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=2 * latent_dim)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d(2 * latent_dim, (in_channels + 2 * latent_dim) // 2, kernel_size=1, stride=1, padding=0)]
        if is_bn:
            layers += [nn.BatchNorm2d(num_features=(in_channels + 2 * latent_dim) // 2)]
        layers += [nn.ReLU()]
        layers += [nn.Conv2d((in_channels + 2 * latent_dim) // 2, in_channels, kernel_size=1, stride=1, padding=0)]

        self.decoder = nn.Sequential(*layers)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


# Feature extractor
class resnet_feature_extractor(torch.nn.Module):
    def __init__(self):
        super(resnet_feature_extractor, self).__init__()
        self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Hook to extract feature maps
        def hook(module, input, output):
            self.features.append(output)

        self.model.layer2[-1].register_forward_hook(hook)
        self.model.layer3[-1].register_forward_hook(hook)

    def forward(self, input):
        self.features = []
        with torch.no_grad():
            _ = self.model(input)
        avg = torch.nn.AvgPool2d(3, stride=1)
        fmap_size = self.features[0].shape[-2]
        resize = torch.nn.AdaptiveAvgPool2d(fmap_size)
        resized_maps = [resize(avg(fmap)) for fmap in self.features]
        patch = torch.cat(resized_maps, 1)
        return patch


def decision_function(segm_map):  

    mean_top_10_values = []

    for map in segm_map:
        # Flatten the tensor
        flattened_tensor = map.reshape(-1)

        # Sort the flattened tensor along the feature dimension (descending order)
        sorted_tensor, _ = torch.sort(flattened_tensor,descending=True)

        # Take the top 10 values along the feature dimension
        mean_top_10_value = sorted_tensor[:10].mean()

        mean_top_10_values.append(mean_top_10_value)

    return torch.stack(mean_top_10_values)






# Load the model
model = FeatCAE(in_channels=1536, latent_dim=100)
model.load_state_dict(torch.load('autoencoder_with_resnet_deep_features.pth', map_location=torch.device('cpu')))
model.eval()

# Define transformations
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# Load an image
# 'dataset/test/good/captured_image_184.jpg'
image_path = 'dataset/test/good/captured_image_1_angle_0.jpg'  # Replace with your image path
image = Image.open(image_path).convert('RGB')
image = transform(image).unsqueeze(0)  # Add batch dimension


# Instantiate the backbone
backbone = resnet_feature_extractor()

with torch.no_grad():
    # Extract features and reconstruct
    features = backbone(image)
    recon = model(features)

# Compute reconstruction error
recon_error = ((features - recon) ** 2).mean(axis=(1)).unsqueeze(0)

# Upscale by bilinear interpolation to match input resolution
segm_map = torch.nn.functional.interpolate(
    recon_error,
    size=(224, 224),
    mode='bilinear'
)

segm_map_unscaled =  ((features-recon)**2).mean(axis=(1))[:,3:-3,3:-3]
anomaly_score = decision_function(segm_map_unscaled)
print("Anomaly score: ", anomaly_score[0].item())

# Convert to NumPy array
segm_map_np = segm_map.squeeze().cpu().numpy()

# Set static color scale (adjust vmin and vmax based on your dataset)
vmin = 0.0   # Minimum value for color scaling
vmax = 0.1   # Maximum value for color scaling

# Plot with static color scaling and colorbar
plt.figure(figsize=(6, 6))
im = plt.imshow(segm_map_np, cmap='jet', vmin=vmin, vmax=vmax)  # Set static scale

# Add colorbar with label
plt.colorbar(im, orientation='vertical', label='Reconstruction Error')

# Display the plot
plt.title(f'Reconstruction Error Map\nAnomaly Score: {anomaly_score[0].item():.4f}')
plt.show()
