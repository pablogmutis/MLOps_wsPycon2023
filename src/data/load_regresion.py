import torch
import torchvision
from torch.utils.data import TensorDataset
import argparse
import wandb

#comment

parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID de la ejecución')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")

def preprocess_mnist_for_regression(dataset):
    """
    Preprocesses MNIST dataset to have (average intensity, label) pairs.
    """
    features = []
    targets = []
    for img, target in dataset:
        # Convert PIL Image to PyTorch tensor, flatten, and calculate average intensity
        img_tensor = torchvision.transforms.ToTensor()(img).flatten()
        avg_intensity = img_tensor.mean().unsqueeze(0) # Shape (1,)
        features.append(avg_intensity)
        targets.append(torch.tensor([float(target)])) # Convert label to float tensor shape (1,)
    features = torch.stack(features) # Shape (N, 1)
    targets = torch.stack(targets)   # Shape (N, 1)
    return features, targets

def load_mnist_for_regression(train_size=.8):
    """
    Loads MNIST and preprocesses it for a simple regression task.
    """
    train_dataset = torchvision.datasets.MNIST(root='./data', train=True, download=True)
    test_dataset = torchvision.datasets.MNIST(root='./data', train=False, download=True)

    train_features, train_targets = preprocess_mnist_for_regression(train_dataset)
    test_features, test_targets = preprocess_mnist_for_regression(test_dataset)

    # Split off a validation set
    train_len = int(len(train_features) * train_size)
    val_features = train_features[train_len:]
    val_targets = train_targets[train_len:]
    train_features = train_features[:train_len]
    train_targets = train_targets[:train_len]

    training_set = TensorDataset(train_features, train_targets)
    validation_set = TensorDataset(val_features, val_targets)
    test_set = TensorDataset(test_features, test_targets)
    datasets = [training_set, validation_set, test_set]
    return datasets

def load_and_log_regression_data():
    with wandb.init(
        project="MLOps-Pycon2023-Regression", # Use the regression project
        name=f"Load Regression Data ExecId-{args.IdExecution}", job_type="load-data") as run:

        datasets = load_mnist_for_regression()
        names = ["training", "validation", "test"]

        raw_data = wandb.Artifact(
            "simple-linear-regression-data", # Use the expected artifact name
            type="dataset",
            description="MNIST data preprocessed for simple linear regression (avg intensity -> label)",
            metadata={"source": "torchvision.datasets.MNIST",
                      "transformation": "average pixel intensity",
                      "sizes": [len(dataset) for dataset in datasets]})

        for name, data in zip(names, datasets):
            with raw_data.new_file(name + ".pt", mode="wb") as file:
                x, y = data.tensors
                torch.save((x, y), file)

        run.log_artifact(raw_data)
        print(f"Artifact 'simple-linear-regression-data' logged to WandB run {run.id}")

# testing
load_and_log_regression_data()
