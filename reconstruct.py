import torch
import numpy as np
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from utils import load_mat, show_image
from train import Model  # assuming Model is defined in train.py, otherwise import from models.py

# ====== Configuration ======
dataset_path = './datasets/Wang/ShapeSpace.mat'
checkpoint_path = './checkpoints/metalattice_model.pth'
x_dim = 2500
im_x = 50
im_y = 50
hidden_dim = 32
latent_dim = 16
model_type = 'CNN'
device = torch.device("cpu")  # Change to "cuda" if you're using GPU

# ====== Load trained model ======
loaded_model = torch.load(checkpoint_path, map_location=device)
loaded_model.eval()

# ====== Load and preprocess dataset ======
mat_data = load_mat(dataset_path)
dataset = mat_data['ShapeSpace']              # shape: (50, 50, N)
dataset = np.transpose(dataset, (2, 0, 1))    # shape: (N, 50, 50)
dataset = dataset.reshape((-1, x_dim))        # shape: (N, 2500)
dataset = dataset.astype(np.float32)
dataset = dataset[:256]                       # optional limit

# ====== Prepare test data ======
_, X_test = train_test_split(dataset, test_size=0.1, random_state=42)
X_test = torch.tensor(X_test).to(torch.float32)
test_loader = DataLoader(X_test, batch_size=1, shuffle=True)

# ====== Reconstruct and visualize one sample ======
with torch.no_grad():
    for x in test_loader:
        x = x.to(device)
        x_hat, _, _ = loaded_model(x)
        break  # only process one sample

# ====== Show original and reconstructed image ======
# Convert to NumPy arrays
original = x[0].view(im_x, im_y).cpu().numpy()
reconstructed = x_hat[0].view(im_x, im_y).cpu().numpy()

# First window: Original
plt.figure()
plt.imshow(original, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Second window: Reconstructed
plt.figure()
plt.imshow(reconstructed, cmap='gray')
plt.title('Reconstructed Image')
plt.axis('off')

# Show both
plt.show()

