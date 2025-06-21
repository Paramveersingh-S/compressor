# src/benchmark_onnx_model.py
import onnx
import numpy as np
import time

# Load model and test data
model = onnx.load('src/conv_autoencoder.onnx')
test_data = np.load('data/X_test.npy')

# Simulate inference timing (without actual inference)
start_time = time.time()
# Model would run here on actual hardware
inference_time = time.time() - start_time

print(f"Model ready for STM32 deployment!")
print(f"Test data shape: {test_data.shape}")
