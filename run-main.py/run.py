# from torch.distributed.pipeline.sync.worker import worker

from ultralytics import YOLO
# from ultralytics import RTDETR
from ultralytics import settings
import torch
if __name__ == '__main__':
    print(settings)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # 加载模型
    model = YOLO("yolov9t.yaml")
    # model = YOLO("yolov9t-EFF.yaml")
    # model = YOLO("yolov9t-EFF-CAM.yaml")
    # model = YOLO("yolov9t-EFF-CAM-Dysample.yaml") 
    # Display model information (optional)
    # model.info()

    # 使用模型
    # results = model.train(resume=True)
    # result = model.train(data="AABmydata.yaml", epochs=200,patience=300,batch=8,imgsz=1472) # 训练模型
    result = model.train(data="AABmydata.yaml", epochs=200,patience=200,batch=4,imgsz=1472) # 训练模型
    # result = model.train(data="AABmydata.yaml", epochs=200,patience=300,batch=8,imgsz=1472,device=1) # 训练模型

    # metrics = model.val()  # 在验证集上评估模型性能
    # metrics.box.map    # map50-95
    # metrics.box.map50  # map50
    # metrics.box.map75  # map75
    # metrics.box.maps   # a list contains map50-95 of each category
    # model = YOLO(r"E:\csv\after_reshoot\B4_1460\weights\best.pt")
    # results = model(r"H:\download\Auto_splitimg-main\Auto_splitimg-main\Chunking and Recovery\images_split",save = False, device=0)
    # results = model(sources, device=0)
    # for result in results:
    #     boxes = result.boxes  # 边界框输出的 Boxes 对象
    #     print(boxes)
    # results = model(r"D:\desktop\use_img",save = True,show_conf=False,conf=0.82,iou=0.3)  # 对图像进行预测
    # model = YOLO('infer_1472.pt')
    # success = model.export(format="onnx", opset=11)  # 将模型导出为 ONNX 格式
    # model = YOLO("ECA.pt")  # 加载预训练模型（建议用于训练）
    # model = YOLO("wiou_eca.pt")  # 加载预训练模型（建议用于训练）
    # results = model(r"D:\\desktop\\use_img\\part\\ori.jpg")  # 对图像进行预测


    #详细显示网络结构
    # x = torch.randn(1, 3, 640, 640)
    # script_model = torch.jit.trace(model.model,x,strict=False)
    # script_model.save("yolov8-wiou.pt")

    # model  path to model file, i.e. yolov8n.pt, yolov8n.yaml
    # data  path to data file, i.e. coco128.yaml
    # epochs
    # patience
    # batch  number of images per batch (-1 for AutoBatch)
    # imgsz  	size of input images as integer
    # device = [0, 1]
    # workers