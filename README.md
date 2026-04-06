A vision-based detection-localization-measurement framework for quantitative size characterization of welding micro-defects
I. Technical Flow of the Integrated Framework
Original high-resolution images
↓
Local feature sub-images
↓
Dataset configuration
↓
Improved YOLOv9 model training
↓
Welding micro-defect detection and classification
↓
Defect spatial localization and coordinate output
↓
Mapping local feature sub-image detection results back to original high-resolution images
↓
Micro-defect dimension measurement
II. Project File Structure
├── run.py                  # Main entry point of the framework
├── AABmydata.yaml          # Dataset configuration file
├── Model Config/           # Improved YOLOv9 model structure library
│   ├── yolov9t.yaml
│   ├── yolov9t-EFF.yaml
│   ├── yolov9t-EFF-CAM.yaml
│   └── yolov9t-EFF-CAM-Dysample.yaml
└── ultralytics/
    ├── nn/modules.py       # Network layer implementation
    ├── models/yolo/detect/train.py  # DetectionTrainer
    ├── utils/              # Data augmentation, loss function, and evaluation tools
    └── engine/             # Training and inference engine
Ⅲ. Description of Core Framework Modules
The improved YOLOv9 model structure is uniformly defined by the .yaml file. 
The framework automatically parses the Backbone, Neck and Head structures, and loads the corresponding network layers from ultralytics/nn/modules.py to complete model construction. 
The training process is automatically scheduled by the DetectionTrainer to achieve data loading, forward inference and loss optimization.
Ⅳ. Environment Configuration
Bash
pip install torch torchvision
pip install ultralytics
pip install opencv-python numpy pillow matplotlib
Ⅴ. Framework Training
from ultralytics import YOLO
import torch
if __name__ == '__main__':
    model = YOLO("Model Config/yolov9t-EFF-CAM-Dysample.yaml")
    result = model.train(
        data="AABmydata.yaml",
        epochs=200,
        patience=200,
        batch=4,
        imgsz=1472
    )    # Run the run.py file.  
Ⅵ. Training Output
best.pt:Optimal weights
last.pt:Final epoch weights
results.png:Curves of loss, mAP, precision and recal


