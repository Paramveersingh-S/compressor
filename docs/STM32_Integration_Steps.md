# STM32 Integration Steps

## Monday Setup (When STM32 Arrives)

### 1. STM32Cube.AI Setup
1. Open STM32CubeMX
2. Load your ONNX model: `src/conv_autoencoder.onnx`
3. Generate C code
4. Copy generated files to your STM32CubeIDE project

### 2. Project Integration
1. Create new STM32CubeIDE project for your Discovery board
2. Copy `firmware/stm32_ecocompress_template.c` as starting point
3. Integrate generated AI files
4. Configure UART for debugging output

### 3. Build and Test
1. Build project
2. Flash to STM32
3. Monitor UART output for results

## Expected Performance
- Inference time: ~5-10ms
- Memory usage: ~20-50KB RAM
- Model size: ~50KB Flash
