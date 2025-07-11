import mlflow
from mlflow.tracking import MlflowClient
import numpy as np
import torch


def get_experiment_runs(experiment_name: str, **kwargs):

  client = MlflowClient()
  experiment = client.get_experiment_by_name(experiment_name)
  runs = client.search_runs(experiment.experiment_id, **kwargs)

  if runs:
    print(f"[INFO] Found {len(runs)} runs")
  else:
    print(f"Found no runs for experiment {experiment_name}")
  
  return runs


def get_latest_run(experiment_name: str):
  print("[WARNING] There's no guarantee that this function actually gets the latest run, I'm just sorting runs by 'start_time'")
  client = MlflowClient()
  experiment = client.get_experiment_by_name(experiment_name)
  runs = client.search_runs(experiment.experiment_id, order_by=['start_time'])
  return runs[0]

def get_latest_run_id(experiment_name: str):
  run = get_latest_run(experiment_name)
  return run.info.run_id

def get_run_description(run_id):
  client = MlflowClient()
  run = client.get_run(run_id)
  return run.data.tags['mlflow.note.content']

def update_run_description(run_id, notes):
    """Add notes to an existing run."""
    client = MlflowClient()
    client.set_tag(run_id, "mlflow.note.content", notes)
    print(f"Added notes to run {run_id}")

def get_input_example(train_dataloader):
  "Use a dataloader to get an example of the input type"
  i = 0
  iterable = iter(train_dataloader)
  x, y = next(iterable)
  return np.asarray(x)

def get_output_example(model, train_dataloader):
  "Use the model and the dataloader to get an example of the output type"
  device = torch.device("cuda" if next(model.parameters()).is_cuda else 'cpu')
  input_example = get_input_example(train_dataloader)
  with torch.no_grad():
    output_example = model(torch.tensor(input_example).to(device)).cpu().numpy()
  return output_example