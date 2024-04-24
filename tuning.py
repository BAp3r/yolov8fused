from ultralytics import YOLO
# Install and update Ultralytics and Ray Tune packages
# pip install -U ultralytics "ray[tune]<=2.9.3"

# # Optionally install W&B for logging
# pip install wandb

# Define a YOLO model
model = YOLO("yolov8custom.pt")

# Run Ray Tune on the model
result_grid = model.tune(data="bdd100k.yaml",
                         space={"lr0": tune.uniform(1e-5, 1e-1)},
                         epochs=50,
                         use_ray=True)