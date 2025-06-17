import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# PARAMETERS
WINDOW_SIZE = 128
NUM_FEATURES = 13
BATCH_SIZE = 64
NUM_EPOCHS = 20
LEARNING_RATE = 0.001

# Load preprocessed data
X_train = np.load('data/X_train.npy')
X_val = np.load('data/X_val.npy')

# Convert to torch tensors and permute to (batch, channels, seq_len)
X_train_t = torch.tensor(X_train, dtype=torch.float32).permute(0, 2, 1)
X_val_t = torch.tensor(X_val, dtype=torch.float32).permute(0, 2, 1)

# Define the 1D convolutional autoencoder
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

# Instantiate model, loss, optimizer
model = ConvAutoencoder(NUM_FEATURES, WINDOW_SIZE)
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Data loaders
train_dataset = TensorDataset(X_train_t, X_train_t)
val_dataset = TensorDataset(X_val_t, X_val_t)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Training loop
train_losses = []
val_losses = []

for epoch in range(NUM_EPOCHS):
    model.train()
    running_loss = 0.0
    for inputs, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
    epoch_loss = running_loss / len(train_loader.dataset)
    train_losses.append(epoch_loss)

    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for inputs, targets in val_loader:
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item() * inputs.size(0)
    val_loss /= len(val_loader.dataset)
    val_losses.append(val_loss)

    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}")

# Save the trained model
torch.save(model.state_dict(), 'src/conv_autoencoder.pth')
print("Model saved to src/conv_autoencoder.pth")
