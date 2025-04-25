import torch
from torch.utils.data import TensorDataset
import os
import argparse
import wandb

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

def preprocess_regression(dataset, normalize=False):
    """
    Preprocesses regression data (features, targets).
    Assumes features have shape (N, 1) and targets have shape (N, 1).
    Normalization can be applied if needed.
    """
    x, y = dataset.tensors

    if normalize:
        # Apply normalization to the features (assuming single feature)
        mean_x = torch.mean(x)
        std_x = torch.std(x)
        if std_x != 0:
            x = (x - mean_x) / std_x
        # You might choose to normalize targets as well depending on your task
        # mean_y = torch.mean(y)
        # std_y = torch.std(y)
        # if std_y != 0:
        #     y = (y - mean_y) / std_y

    return TensorDataset(x, y)

def preprocess_and_log_regression(steps):
    with wandb.init(project="MLOps-Pycon2023-Regression", # Use the regression project
                    name=f"Preprocess Regression Data ExecId-{args.IdExecution}", job_type="preprocess-data") as run:
        processed_data = wandb.Artifact(
            "simple-linear-regression-preprocessed", # New artifact name for preprocessed regression data
            type="dataset",
            description="Preprocessed data for simple linear regression",
            metadata=steps)

        # ✔️ declare which artifact we'll be using
        raw_data_artifact = run.use_artifact('simple-linear-regression-data:latest')

        # 📥 if need be, download the artifact
        raw_dataset_dir = raw_data_artifact.download(root="./data/regression_artifacts/")

        for split in ["training", "validation", "test"]:
            raw_split = read_regression_artifact(raw_dataset_dir, split)
            processed_dataset = preprocess_regression(raw_split, **steps)

            with processed_data.new_file(split + ".pt", mode="wb") as file:
                x, y = processed_dataset.tensors
                torch.save((x, y), file)

        run.log_artifact(processed_data)
        print(f"Artifact 'simple-linear-regression-preprocessed' logged to WandB run {run.id}")

def read_regression_artifact(data_dir, split):
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)

steps_regression = {"normalize": True} # Define preprocessing steps for regression

preprocess_and_log_regression(steps_regression)
