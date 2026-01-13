"""
Module for logging utilities.
"""

import csv
import datetime
import os

import torch
import yaml
from pydantic import BaseModel

def create_run_dir(base_dir="./runs", trial_name=None):
    """
    Create a run directory for storing outputs, organized with timestamps 
    or trial identifiers.

    Args:
        base_dir (str): Base directory for runs (default "./runs")
        trial_name (str or None): Optional trial name (e.g., Ray Tune trial)

    Returns:
        str: Path to the created run directory

    Directory Structure:
        If trial_name is provided:
            runs/<trial_name>/
        Otherwise:
            runs/YYYY-MM-DD_HH-MM-SS/
    """

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if trial_name:
        run_dir = os.path.join(base_dir, trial_name)
        os.makedirs(run_dir, exist_ok=True)
    else:
        run_dir = os.path.join(base_dir, timestamp)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir

def save_config(config, run_dir):
    """
    Save a Pydantic configuration object as a YAML file.

    Args:
        config (BaseModel): Pydantic config object
        run_dir (str): Directory where config.yaml will be saved

    Returns:
        str: Full path to the saved YAML file
    """
    path = os.path.join(run_dir, "config.yaml")

    # Convert Pydantic object → nested dict (safe for YAML)
    if isinstance(config, BaseModel):
        config_dict = config.model_dump()
    else:
        config_dict = config  # fallback, e.g. dict input

    with open(path, "w") as f:
        yaml.safe_dump(config_dict, f, sort_keys=False)

    return path


def save_model_summary(model, run_dir):
    """
    Save a model architecture summary to a text file in the run directory.

    Args:
        model (torch.nn.Module): Model to summarize
        run_dir (str): Directory where model.txt will be saved

    Returns:
        str: Path to the saved text file
    """
    
    path = os.path.join(run_dir, "model.txt")
    with open(path, "w") as f:
        f.write(str(model))
    return path

class CSVLogger:
    """
    Minimal CSV logger for recording training or validation metrics.

    Example usage:
        logger = CSVLogger(run_dir)
        logger.log({"epoch": 1, "train_loss": 0.5, "val_loss": 0.4})
        logger.close()

    Attributes:
        path (str): Path to the CSV file
        file (TextIO): File handle
        writer (csv.DictWriter): CSV writer object
    """

    def __init__(self, run_dir):
        """
        Initialize the CSV logger.
        run_dir: Directory where metrics.csv will be stored.
        """
        self.path = os.path.join(run_dir, "metrics.csv")
        self.file = open(self.path, "w", newline="", encoding="utf-8")
        self.writer = None

    def log(self, metrics: dict):
        """
        Log a dictionary of metrics to the CSV file.

        Args:
            metrics (dict): Dictionary where keys are column names and values are metric values
        """
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=metrics.keys())
            self.writer.writeheader()
        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        self.file.close()

def save_checkpoint(model, run_dir, filename="checkpoint.pt"):
    """
    Save a PyTorch model checkpoint (state_dict).

    Args:
        model (torch.nn.Module): Model to save
        run_dir (str): Directory where checkpoint will be stored
        filename (str): Name of the checkpoint file (default "checkpoint.pt")

    Returns:
        str: Path to the saved checkpoint
    """
    path = os.path.join(run_dir, filename)
    torch.save(model.state_dict(), path)
    return path

def append_global_trial_summary(summary: dict, base_dir):
    """
    Append a summary of a trial (hyperparameters and metrics) to a global CSV file.

    If the file does not exist, a new CSV file with a header is created.

    Args:
        summary (dict): Dictionary of trial results (e.g., metrics, hyperparameters)
        base_dir (str): Directory where grid_summary.csv will be stored
    """
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, "grid_summary.csv")

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)
