import torch
import torch.nn as nn
import os
import argparse
import wandb

#comentario
parser = argparse.ArgumentParser()
parser.add_argument('--IdExecution', type=str, help='ID de la ejecución')
args = parser.parse_args()

if args.IdExecution:
    print(f"IdExecution: {args.IdExecution}")
else:
    args.IdExecution = "testing console"

# Check if the directory "./model" exists
if not os.path.exists("./model"):
    # If it doesn't exist, create it
    os.makedirs("./model")

# Data parameters testing para regresión lineal simple
input_dim = 1  # Regresión lineal simple tiene una sola característica de entrada
output_dim = 1 # Regresión lineal simple predice un solo valor

# Definición de un modelo lineal simple
class LinearRegressor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressor, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.linear(x)

def build_model_and_log_regression(config, model, model_name="SimpleLinearRegression", model_description="Simple Linear Regression Model"):
    with wandb.init(project="MLOps-Pycon2023-Regression",
                    name=f"initialize-{model_name}-ExecId-{args.IdExecution}",
                    job_type="initialize-model", config=config) as run:
        config = wandb.config

        model_artifact = wandb.Artifact(
            model_name, type="model",
            description=model_description,
            metadata=dict(config))

        name_artifact_model = f"initialized_{model_name}.pth"

        torch.save(model.state_dict(), f"./model/{name_artifact_model}")
        # ➕ another way to add a file to an Artifact
        model_artifact.add_file(f"./model/{name_artifact_model}")

        wandb.save(name_artifact_model)

        run.log_artifact(model_artifact)

# Configuración del modelo para regresión lineal simple
regression_config = {"input_dim": input_dim,
                       "output_dim": output_dim}

# Instanciación del modelo de regresión lineal
linear_model = LinearRegressor(**regression_config)

# Construcción y registro del modelo
build_model_and_log_regression(regression_config, linear_model, "SimpleLinearRegression", "Simple Linear Regression Model")
