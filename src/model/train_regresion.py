import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import wandb
import argparse
import os

# comment
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID of the execution')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Device configuration
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("Device: ", device)

# Definición de la clase LinearRegressor AL PRINCIPIO del archivo
class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)


def read_regression(data_dir, split):
    """
    Read regression data from a directory and return a TensorDataset object.
    Assumes data is saved as (features, targets) in a .pt file.
    Features should have shape (N, 1) for simple linear regression.
    Targets should have shape (N, 1).

    Args:
    - data_dir (str): The directory where the data is stored.
    - split (str): The name of the split to read (e.g. "train", "valid", "test").

    Returns:
    - dataset (TensorDataset): A TensorDataset object containing the regression data.
    """
    filename = split + ".pt"
    x, y = torch.load(os.path.join(data_dir, filename))
    return TensorDataset(x, y)


def train_regression(model, train_loader, valid_loader, config):
    optimizer = getattr(torch.optim, config.optimizer)(model.parameters(), lr=config.learning_rate)
    criterion = nn.MSELoss()  # Mean Squared Error for regression
    model.train()
    example_ct = 0
    for epoch in range(config.epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device).float(), target.to(device).float() # Ensure float for regression
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            example_ct += len(data)

            if batch_idx % config.batch_log_interval == 0:
                print('Train Epoch: {} [{}/{} ({:.0%})]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    batch_idx / len(train_loader), loss.item()))

                train_log_regression(loss, example_ct, epoch)

        # evaluate the model on the validation set at each epoch
        val_loss = test_regression(model, valid_loader, criterion)
        test_log_regression(val_loss, example_ct, epoch)


def test_regression(model, test_loader, criterion):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)
            test_loss += criterion(output, target).item() * data.size(0)  # sum up batch loss

    test_loss /= len(test_loader.dataset)
    return test_loss


def train_log_regression(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "train/loss": loss}, step=example_ct)
    print(f"Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")


def test_log_regression(loss, example_ct, epoch):
    loss = float(loss)
    wandb.log({"epoch": epoch, "validation/loss": loss}, step=example_ct)
    print(f"Validation Loss after " + str(example_ct).zfill(5) + f" examples: {loss:.3f}")

def evaluate_regression(model, test_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device).float(), target.to(device).float()
            output = model(data)
            total_loss += criterion(output, target).item() * data.size(0)

    avg_loss = total_loss / len(test_loader.dataset)
    return avg_loss

def train_and_log_regression(config, experiment_id='99'):
    with wandb.init(
        project="MLOps-Pycon2023-Regression",
        name=f"Train-LinearReg-ExecId-{args.IdExecution}-ExpId-{experiment_id}",
        job_type="train-model", config=config) as run:
        config = wandb.config
        data = run.use_artifact('simple-linear-regression-data:latest') # Assuming you have regression data artifact
        data_dir = data.download()

        training_dataset = read_regression(data_dir, "training")
        validation_dataset = read_regression(data_dir, "validation")

        train_loader = DataLoader(training_dataset, batch_size=config.batch_size)
        validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)

        model_artifact = run.use_artifact("simple-linear-regression:latest") # Use the linear regression model artifact
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "initialized_simple-linear-regression.pth")
        model_config = model_artifact.metadata
        config.update(model_config)

        # Instancia la clase LinearRegressor directamente ya que está definida aquí
        model = LinearRegressor(**model_config)
        model.load_state_dict(torch.load(model_path))
        model = model.to(device)

        train_regression(model, train_loader, validation_loader, config)

        model_artifact = wandb.Artifact(
            "trained-linear-regression-model", type="model",
            description="Trained simple linear regression model",
            metadata=dict(model_config))

        torch.save(model.state_dict(), "trained_linear_regression_model.pth")
        model_artifact.add_file("trained_linear_regression_model.pth")
        wandb.save("trained_linear_regression_model.pth")

        run.log_artifact(model_artifact)

    return model


def evaluate_and_log_regression(experiment_id='99', config=None):
    with wandb.init(project="MLOps-Pycon2023-Regression", name=f"Eval-LinearReg-ExecId-{args.IdExecution}-ExpId-{experiment_id}", job_type="eval-model", config=config) as run:
        data = run.use_artifact('simple-linear-regression-data:latest') # Assuming you have regression data artifact
        data_dir = data.download()
        testing_set = read_regression(data_dir, "test")
        test_loader = DataLoader(testing_set, batch_size=128, shuffle=False)

        model_artifact = run.use_artifact("trained-linear-regression-model:latest")
        model_dir = model_artifact.download()
        model_path = os.path.join(model_dir, "trained_linear_regression_model.pth")
        model_config = model_artifact.metadata

        # Instancia la clase LinearRegressor directamente ya que está definida aquí
        model = LinearRegressor(**model_config)
        model.load_state_dict(torch.load(model_path))
        model.to(device)

        criterion = nn.MSELoss()
        avg_loss = evaluate_regression(model, test_loader, criterion)

        run.summary.update({"loss": avg_loss})
        wandb.log({"test/loss": avg_loss})
        print(f"Test Loss: {avg_loss:.6f}")
