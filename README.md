# EcoCompress

**EcoCompress** is an open-source, low-power AI/ML-based data compression toolkit designed for resource-constrained edge devices and IoT sensors. It enables efficient, adaptive, and high-fidelity compression of time-series sensor data using lightweight neural networks, with a deployment-ready workflow for STM32 microcontrollers.

---

## 🚀 Features

- **Lightweight 1D Convolutional Autoencoder** for time-series sensor data
- **High compression ratios** with tunable fidelity (MSE/PSNR metrics)
- **Preprocessing pipeline**: normalization, windowing, train/val/test split
- **Model optimization**: quantization, ONNX export, TensorFlow conversion
- **Deployment-ready**: STM32Cube.AI and STM32CubeIDE integration for STM32 MCUs
- **Clear, modular Python scripts** for every step

---

## 📂 Project Structure

EcoCompress/
├── data/ # Raw and processed datasets (.csv, .npy)
├── src/ # All Python scripts
│ ├── preprocess_window.py
│ ├── train_autoencoder_tuned.py
│ ├── evaluate_autoencoder.py
│ ├── optimize_model.py
│ ├── onnx_to_tf.py
├── notebooks/ # (Optional) Jupyter notebooks for EDA/prototyping
├── docs/ # Documentation
├── firmware/ # STM32 deployment files and code
├── tests/ # Unit tests (optional)
├── requirements.txt # Python dependencies
├── README.md


---

## 🛠️ Quick Start

### 1. **Clone the Repository**
git clone https://github.com/Paramveersingh-S/compressor.git
cd compressor

### 2. **Install Dependencies**
python -m pip install -r requirements.txt

### 3. **Preprocess Data**
Place your sensor data CSV (e.g., `AirQualityUCI.csv`) in `data/`, then run:
python src/preprocess_window.py

### 4. **Train the Model**
python src/train_autoencoder_tuned.py

### 5. **Evaluate Model Fidelity**
python src/evaluate_autoencoder.py

### 6. **Optimize and Export Model**
python src/optimize_model.py

### 7. **Convert ONNX to TensorFlow (for STM32Cube.AI)**
python src/onnx_to_tf.py


### 8. **Deploy to STM32**
- Use STM32Cube.AI to import the ONNX or TensorFlow model.
- Generate and integrate C code into your STM32CubeIDE project.
- Flash and run on your STM32 Discovery board!

---

## 📊 Results & Benchmarks

- **Compression Ratio:** Achieves up to 10:1 (lossy, tunable)
- **Fidelity:** PSNR > 21 dB (can be increased with tuning)
- **Model Size:** < 50 KB after quantization (STM32-ready)
- **Supported Hardware:** STM32 Discovery and Nucleo boards

---

## 📖 Documentation

- [STM32Cube.AI User Guide](https://www.st.com/en/development-tools/stm32cubeai.html)
- [ONNX to TensorFlow Conversion](https://github.com/onnx/onnx-tensorflow)
- [Project Wiki](docs/) (coming soon)

---

## 🤝 Contributing

Contributions, bug reports, and feature requests are welcome!
- Fork the repo and submit a pull request.
- Open an issue for questions or suggestions.

---

## 📄 License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

---

## 🙏 Acknowledgements

- [UCI Air Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Air+Quality)
- [PyTorch](https://pytorch.org/), [ONNX](https://onnx.ai/), [STM32Cube.AI](https://www.st.com/en/development-tools/stm32cubeai.html)
- Inspired by CERN’s Baler project and TinyML best practices.

---
