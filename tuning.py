from ultralytics import YOLO

# 用自定义的数据集和结构从头训练
# 导入结构yaml


# model = YOLO('yolov8s.pt')
model.tune(data="bdd100k.yaml", epochs=20, iterations=300, optimizer="Lion", plots=False, save=False, val=False)
