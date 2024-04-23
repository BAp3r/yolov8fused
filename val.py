import warnings

warnings.filterwarnings("ignore")
from ultralytics import YOLO

if __name__ == "__main__":
    model = YOLO(r"C:\PostGraduate\Fish\YOLO\ultralytics\runs\train\yolov8-pose\weights\best.pt")
    model.val(
        data="ultralytics/cfg/datasets/data-pose.yaml",
        split="val",
        imgsz=640,
        batch=16,
        # rect=False,
        # save_json=True,
        project="runs/val",
        name="yolov8-pose",
    )
