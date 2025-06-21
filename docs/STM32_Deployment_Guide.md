
# STM32Cube.AI Deployment Guide for EcoCompress

## Prerequisites
- STM32CubeIDE installed
- STM32Cube.AI v10.0+ installed
- STM32 Discovery board
- USB cable for programming

## Step 1: Import Model to STM32Cube.AI

1. **Open STM32CubeMX**
2. **Navigate to Additional Software > STM32Cube.AI**
3. **Click "Add network"**
4. **Select your ONNX model**: `src/conv_autoencoder.onnx`
5. **Set network name**: `ecocompress_model`

## Step 2: Analyze Model

1. **Click "Analyze"** - STM32Cube.AI will:
   - Check model compatibility
   - Estimate RAM/Flash usage
   - Calculate inference time
   - Suggest optimizations

2. **Review Analysis Results**:
   - Flash usage should be < 512KB for most STM32s
   - RAM usage should be < 64KB for inference
   - Inference time target: < 10ms

## Step 3: Validate Model

1. **Click "Validate"**
2. **Choose validation method**:
   - Use your test data from `data/X_test.npy`
   - Or generate random inputs
3. **Verify accuracy** matches your Python model

## Step 4: Generate Code

1. **Click "Generate Code"**
2. **STM32Cube.AI will create**:
   - `network.c/h` - Neural network implementation
   - `network_data.c` - Model weights and biases
   - `app_x-cube-ai.c` - Application template

## Step 5: Integrate into STM32CubeIDE

1. **Create new STM32CubeIDE project** for your Discovery board
2. **Copy generated AI files** to your project
3. **Add preprocessing code** (windowing, normalization)
4. **Add postprocessing code** (denormalization, output handling)

## Step 6: Build and Flash

1. **Build project** in STM32CubeIDE
2. **Flash to your STM32 board**
3. **Test with sample data**

## Estimated Performance Targets

- **Model Size**: ~50KB Flash
- **RAM Usage**: ~20KB during inference  
- **Inference Time**: ~5ms @ 80MHz
- **Power Consumption**: ~10mA during inference

## Troubleshooting

- If model too large: Enable quantization in STM32Cube.AI
- If inference too slow: Increase CPU frequency or optimize model
- If accuracy poor: Check data preprocessing pipeline

