import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

# PARAMETERS
WINDOW_SIZE = 128
NUM_FEATURES = 13
MODEL_PATH = 'src/conv_autoencoder.pth'

# Load test data
X_test = np.load('data/X_test.npy')
X_test_t = torch.tensor(X_test, dtype=torch.float32).permute(0, 2, 1)

# Define the same autoencoder architecture as before
class ConvAutoencoder(nn.Module):
    def __init__(self, num_features, window_size):
        super(ConvAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(num_features, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(16, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True),
            nn.Conv1d(8, 4, kernel_size=3, stride=2, padding=1),
            nn.ReLU(True)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(4, 8, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(8, 16, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(True),
            nn.ConvTranspose1d(16, num_features, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# Load model
model = ConvAutoencoder(NUM_FEATURES, WINDOW_SIZE)
model.load_state_dict(torch.load(MODEL_PATH))
model.eval()

# Evaluate on test data
with torch.no_grad():
    X_test_recon = model(X_test_t).cpu().numpy()

# Calculate MSE and PSNR
mse = np.mean((X_test - np.transpose(X_test_recon, (0,2,1)))**2)
psnr = 10 * np.log10(1 / mse)
print(f"Test MSE: {mse:.6f}")
print(f"Test PSNR: {psnr:.2f} dB")

# Visualize a few reconstructions
idx = 0  # change for different samples
plt.figure(figsize=(12, 6))
plt.plot(X_test[idx, :, 0], label='Original')
plt.plot(X_test_recon[idx, 0, :], label='Reconstructed')
plt.title('Original vs Reconstructed (First Feature)')
plt.xlabel('Time Step')
plt.ylabel('Normalized Value')
plt.legend()
plt.show()

# Estimate compression ratio
# Calculate size of original vs. encoded representation
original_size = X_test.nbytes  # bytes
# For simplicity, estimate encoded size as output of encoder (latent space)
with torch.no_grad():
    latent = model.encoder(X_test_t)
encoded_size = latent.numpy().nbytes
compression_ratio = original_size / encoded_size
print(f"Estimated Compression Ratio: {compression_ratio:.2f}:1")
