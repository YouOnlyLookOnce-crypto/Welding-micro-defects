# from torch.distributed.pipeline.sync.worker import worker

from ultralytics import YOLO
# from ultralytics import RTDETR
from ultralytics import settings
import torch

if __name__ == '__main__':
    print(settings)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = YOLO("yolov9t.yaml")
    # model = YOLO("yolov9t-EFF.yaml")
    # model = YOLO("yolov9t-EFF-CAM.yaml")
    # model = YOLO("yolov9t-EFF-CAM-Dysample.yaml") 

    # Display model information (optional)
    # model.info()

    # Model usage
    # results = model.train(resume=True)
    
    # Train the model
    result = model.train(data="AABmydata.yaml", epochs=200, patience=200, batch=4, imgsz=1472) 
    # result = model.train(data="AABmydata.yaml", epochs=200, patience=300, batch=8, imgsz=1472, device=1) 

    # Evaluate model performance on the validation set
    # metrics = model.val()  
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list containing map50-95 for each category

    # Inference examples
    # model = YOLO(r"E:\csv\after_reshoot\B4_1460\weights\best.pt")
    # results = model(r"H:\download\Auto_splitimg-main\Auto_splitimg-main\Chunking and Recovery\images_split", save=False, device=0)
    # for result in results:
    #     boxes = result.boxes  # Boxes object for bounding box outputs
    #     print(boxes)
    
    # Perform prediction on images
    # results = model(r"D:\desktop\use_img", save=True, show_conf=False, conf=0.82, iou=0.3)  
    
    # Export the model
    # model = YOLO('infer_1472.pt')
    # success = model.export(format="onnx", opset=11)  # Export model to ONNX format
    
    # Load pre-trained models (recommended for training)
    # model = YOLO("ECA.pt")  
    # model = YOLO("wiou_eca.pt")  
    
    # Single image prediction
    # results = model(r"D:\\desktop\\use_img\\part\\ori.jpg")  

    # Detailed network structure visualization/saving
    # x = torch.randn(1, 3, 640, 640)
    # script_model = torch.jit.trace(model.model, x, strict=False)
    # script_model.save("yolov8-wiou.pt")

    # Argument descriptions:
    # model:   path to model file, e.g., yolov8n.pt, yolov8n.yaml
    # data:    path to dataset configuration file, e.g., coco128.yaml
    # epochs:  number of training iterations
    # patience: epochs to wait without improvement before early stopping
    # batch:   number of images per batch (-1 for AutoBatch)
    # imgsz:   size of input images as an integer
    # device:  hardware device (e.g., device=0 or device=[0, 1])
    # workers: number of worker threads for data loading
