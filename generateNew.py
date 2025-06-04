import torch
import numpy as np
import matplotlib.pyplot as plt

from utils import show_image
from train import Model  # or from models import Model if needed

# ====== Configuration ======
checkpoint_path = './checkpoints/metalattice_model.pth'
latent_dim = 16
im_x, im_y = 50, 50
device = torch.device("cpu")  # use "cuda" if you're on a GPU

# ====== Load trained model ======
model = torch.load(checkpoint_path, map_location=device)
model.eval()

# ====== Generate new images from random latent vectors ======
num_samples = 5
with torch.no_grad():
    noise = torch.randn(num_samples, latent_dim).to(device)
    generated = model.Decoder(noise)

    for i in range(num_samples):
        img = generated[i].view(im_x, im_y).cpu().numpy()
        plt.figure()
        plt.imshow(img, cmap='gray')
        plt.title(f"Generated Sample #{i+1}")
        plt.axis('off')

plt.show()
