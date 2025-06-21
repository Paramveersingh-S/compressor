# src/validate_onnx_for_stm32.py
import onnx
import numpy as np
import os

def validate_model_for_stm32():
    """Validate ONNX model for STM32Cube.AI compatibility"""
    
    print("🔍 Validating ONNX model for STM32Cube.AI...")
    
    # Load and validate ONNX model
    try:
        onnx_model = onnx.load('src/conv_autoencoder.onnx')
        onnx.checker.check_model(onnx_model)
        print("✅ ONNX model structure is valid")
    except Exception as e:
        print(f"❌ ONNX validation failed: {e}")
        return False
    
    # Check model details
    print("\n📊 Model Information:")
    print(f"Model inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"Model outputs: {[out.name for out in onnx_model.graph.output]}")
    
    # Get input/output shapes
    for inp in onnx_model.graph.input:
        shape = [dim.dim_value for dim in inp.type.tensor_type.shape.dim]
        print(f"Input '{inp.name}' shape: {shape}")
    
    for out in onnx_model.graph.output:
        shape = [dim.dim_value for dim in out.type.tensor_type.shape.dim]
        print(f"Output '{out.name}' shape: {shape}")
    
    # Check model size
    model_size_kb = os.path.getsize('src/conv_autoencoder.onnx') / 1024
    print(f"Model file size: {model_size_kb:.2f} KB")
    
    # STM32 compatibility checks
    print("\n🎯 STM32 Compatibility Assessment:")
    if model_size_kb < 512:  # 512KB is reasonable for most STM32s
        print("✅ Model size suitable for STM32 deployment")
    else:
        print("⚠️  Model size might be large for some STM32 variants")
    
    # Check for supported operations
    ops = set(node.op_type for node in onnx_model.graph.node)
    print(f"Operations used: {sorted(ops)}")
    
    # STM32Cube.AI typically supports these ops well
    supported_ops = {'Conv', 'ConvTranspose', 'Relu', 'Sigmoid', 'Add', 'Reshape'}
    unsupported_ops = ops - supported_ops
    
    if not unsupported_ops:
        print("✅ All operations are typically supported by STM32Cube.AI")
    else:
        print(f"⚠️  Check these operations in STM32Cube.AI: {unsupported_ops}")
    
    return True

if __name__ == "__main__":
    validate_model_for_stm32()
