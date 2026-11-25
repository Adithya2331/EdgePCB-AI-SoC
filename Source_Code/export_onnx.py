pip install ultralytics onnx onnxslim
yolo export model=pcb.pt format=onnx imgsz=640 simplify=True

from ultralytics import YOLO

model = YOLO("pcb.pt")
model.export(format="onnx", imgsz=640, simplify=True)

import torch
from your_model_file import YourPCBModel  # Import your model architecture

# 1. Load model architecture and weights
model = YourPCBModel()
checkpoint = torch.load("pcb.pt", map_location="cpu")

# If your file is a full training checkpoint, extract the state dict first:
# checkpoint = checkpoint["model_state_dict"]
model.load_state_dict(checkpoint)
model.eval()

# 2. Define a dummy input matching (batch_size, channels, height, width)
dummy_input = torch.randn(1, 3, 640, 640)

# 3. Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "pcb.onnx",
    export_params=True,
    opset_version=17,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
)
