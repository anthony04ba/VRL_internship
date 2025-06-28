import onnxruntime as ort #import onnx runtime or backend 
import numpy as np
import torch

# Load ONNX model
onnx_model_path = "dinov2.onnx"#the onnx mode
session = ort.InferenceSession(onnx_model_path)#instantiate an inference session

# Get input and output names
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Dummy input: same shape as training (1, 3, 512, 960)
dummy_input = torch.randn(1, 3, 512, 960).numpy().astype(np.float32)

# Run inference
outputs = session.run([output_name], {input_name: dummy_input})#run the inference session

# Print shape of output
print(f"ONNX output shape: {outputs[0].shape}")  # Should be (1, 1920, 768) for ViT patch tokens
