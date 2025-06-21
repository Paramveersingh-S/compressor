# src/prepare_benchmark_data.py
import numpy as np
import time

def prepare_test_data_for_stm32():
    """Prepare test data and benchmarking suite for STM32"""
    
    print("🔧 Preparing benchmark data for STM32 testing...")
    
    # Load test data
    X_test = np.load('data/X_test.npy')
    
    # Create smaller test samples for MCU validation
    test_samples = X_test[:10]  # First 10 samples
    
    # Save as C header file for STM32
    generate_c_test_data(test_samples)
    
    # Save as binary file for easy loading
    test_samples.astype(np.float32).tofile('data/stm32_test_data.bin')
    
    print(f"✅ Generated {len(test_samples)} test samples")
    print(f"✅ Test data shape: {test_samples.shape}")
    print(f"✅ Files created:")
    print(f"   - data/stm32_test_data.bin")
    print(f"   - src/test_data.h")

def generate_c_test_data(test_samples):
    """Generate C header file with test data"""
    
    header_content = f"""
#ifndef TEST_DATA_H
#define TEST_DATA_H

#define NUM_TEST_SAMPLES {test_samples.shape[0]}
#define WINDOW_SIZE {test_samples.shape[1]}
#define NUM_FEATURES {test_samples.shape[2]}

// Test input data (normalized)
static const float test_input_data[NUM_TEST_SAMPLES][WINDOW_SIZE][NUM_FEATURES] = {{
"""
    
    for i, sample in enumerate(test_samples):
        header_content += "  {\n"
        for j, window in enumerate(sample):
            header_content += "    {"
            header_content += ", ".join(f"{val:.6f}f" for val in window)
            header_content += "}"
            if j < len(sample) - 1:
                header_content += ","
            header_content += "\n"
        header_content += "  }"
        if i < len(test_samples) - 1:
            header_content += ","
        header_content += "\n"
    
    header_content += """};

#endif // TEST_DATA_H
"""
    
    with open('src/test_data.h', 'w') as f:
        f.write(header_content)

if __name__ == "__main__":
    prepare_test_data_for_stm32()
