# src/optimize_model.py

import torch
from train_autoencoder_tuned import ConvAutoencoder, NUM_FEATURES, WINDOW_SIZE

# 1. Load the trained model
model = ConvAutoencoder(NUM_FEATURES, WINDOW_SIZE)
model.load_state_dict(torch.load('src/conv_autoencoder_tuned.pth'))
model.eval()

# 2. Apply dynamic quantization
quantized_model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Conv1d, torch.nn.ConvTranspose1d, torch.nn.Linear}, dtype=torch.qint8
)
torch.save(quantized_model.state_dict(), 'src/conv_autoencoder_quantized.pth')
print("Quantized model saved to src/conv_autoencoder_quantized.pth")

# 3. Export to ONNX
dummy_input = torch.randn(1, NUM_FEATURES, WINDOW_SIZE)
torch.onnx.export(
    model, dummy_input, "src/conv_autoencoder.onnx",
    input_names=['input'], output_names=['output'], opset_version=11
)
print("Model exported to src/conv_autoencoder.onnx")

# 4. (Optional) Print model size comparison
import os
def get_file_size(path):
    size = os.path.getsize(path) / 1024
    return f"{size:.2f} KB"

print("File sizes:")
print(f"  Original: {get_file_size('src/conv_autoencoder_tuned.pth')}")
print(f"  Quantized: {get_file_size('src/conv_autoencoder_quantized.pth')}")
print(f"  ONNX: {get_file_size('src/conv_autoencoder.onnx')}")
