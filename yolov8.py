from ultralytics import YOLO

import os

model = YOLO("/yolov8n.pt")  # 加载预训练模型（建议用于训练）

if __name__ == '__main__':
    path = os.getcwd() + "/mydata.yaml"
    # 单GPU训练
    results = model.train(data=path, epochs=3, imgsz=640)  # 训练模型
    # 处理结果列表
    for result in results:
        boxes = result.boxes  # 用于边界框输出的 Boxes 对象
        result.show()   # 在屏幕上显示结果