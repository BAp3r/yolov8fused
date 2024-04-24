import pandas as pd
from ray.tune import ExperimentAnalysis, tune

from ultralytics import YOLO


def run_tuning():
    """Runs Ray Tune to tune the YOLO model and saves the results to a CSV file."""
    # Define a YOLO model
    model = YOLO("yolov8custom.pt")

    # Run Ray Tune on the model
    result_grid = model.tune(data="bdd100k.yaml", space={"lr0": tune.uniform(1e-5, 1e-1)}, epochs=20, use_ray=True)

    analysis = ExperimentAnalysis(result_grid)
    best_result = analysis.get_best_trial(metric="performance", mode="max")
    print(f"Best configuration: {best_result.config}")
    print(f"Best performance: {best_result.last_result['performance']}")

    # Convert results to pandas DataFrame
    df = pd.DataFrame(result_grid)

    # Save to CSV
    df.to_csv("advice.csv", index=False)  # replace 'root_directory' with your actual root directory


run_tuning()
