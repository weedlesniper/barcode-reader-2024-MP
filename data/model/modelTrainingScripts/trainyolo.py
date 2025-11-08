import os

if __name__ == '__main__':
    os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'
    
    from ultralytics import YOLO

    model = YOLO("yolov8n-obb.yaml") 
    model = YOLO("yolov8n-obb.pt") 
    model = YOLO("yolov8n-obb.yaml").load("yolov8n.pt")  

    results = model.train(data='A:/unicode/LuniWork/MP/luc_20188193/data/modelv2/data.yaml', epochs=100, imgsz=640)
