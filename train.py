import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO("ultralytics/cfg/models/v8/yolov8-RepViT-BiFPN.yaml")
    # model.load('yolov8n.pt')

    # model = YOLO('ultralytics/cfg/models/v8/yolov8-pose.yaml')   # 关键点检测
    # model.load('yolov8n-pose.pt')

    model.train(
        data="ultralytics/cfg/datasets/bdd100k.yaml",
        epochs=100,
        batch=2,
        workers=2,
        project="runs/train",
        name="yolov8-debug",
        amp=True,
        cache=True,
        imgsz=640,
        # cfg='ultralytics/cfg/cfg-low.yaml',
        # resume='',
        # single_cls=True,
        close_mosaic=10,
        device="0",
        optimizer="Lion",
        seed=0,
        deterministic=True,
        # 50步保存一次
        save_period=50,
    )
