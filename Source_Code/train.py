!pip install ultralytics
import os
from ultralytics import YOLO

# 1. Automatically locate the data.yaml file in the Kaggle input directory
yaml_path = None
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        if filename == 'data.yaml':
            yaml_path = os.path.join(dirname, filename)
            print(f"Found data.yaml at: {yaml_path}")
            break
if yaml_path:
    # 2. Load a pre-trained YOLOv8 model
    # 'yolov8n.pt' is the Nano model. You can also use 'yolov8s.pt' (Small) for higher accuracy.
    model = YOLO('yolov8n.pt') 

    # 3. Train the model on your PCB dataset
    print("Starting YOLOv8 training...")
    results = model.train(
        data=yaml_path,
        epochs=50,          # Number of training loops (increase to 100+ for better accuracy later)
        imgsz=640,          # Standard image size for YOLO
        batch=16,           # Batch size (reduce to 8 if you hit GPU memory limits)
        project='/kaggle/working/pcb_defect_project', # Saves results to the working directory
        name='yolov8_run1' 
    )
    
    print("Training complete! Weights saved in /kaggle/working/pcb_defect_project/yolov8_run1/weights/")
else:
    print("Error: Could not find data.yaml. Please verify the dataset structure in the Input panel.")
    
