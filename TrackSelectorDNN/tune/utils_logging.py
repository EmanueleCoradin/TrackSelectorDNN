import os
import yaml
import csv
import json
import datetime
import torch

def create_run_dir(base_dir="./runs", trial_name=None):
    """
    Create a run directory organized as:
        runs/trial_00001/2025-10-30_14-20-05/
    If not under Ray, just creates:
        runs/2025-10-30_14-20-05/
    """
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    if trial_name:
        trial_dir = os.path.join(base_dir, trial_name)
        os.makedirs(trial_dir, exist_ok=True)
        run_dir = os.path.join(trial_dir, timestamp)
    else:
        run_dir = os.path.join(base_dir, timestamp)

    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def save_config(config_dict, run_dir):
    """Save the config dictionary as YAML."""
    path = os.path.join(run_dir, "config.yaml")
    with open(path, "w") as f:
        yaml.safe_dump(config_dict, f)
    return path


def save_model_summary(model, run_dir):
    """Save model architecture summary as text."""
    path = os.path.join(run_dir, "model.txt")
    with open(path, "w") as f:
        f.write(str(model))
    return path


class CSVLogger:
    """Minimal CSV logger for metrics (epoch, train_loss, val_loss, etc.)."""
    def __init__(self, run_dir):
        self.path = os.path.join(run_dir, "metrics.csv")
        self.file = open(self.path, "w", newline="")
        self.writer = None

    def log(self, metrics: dict):
        if self.writer is None:
            self.writer = csv.DictWriter(self.file, fieldnames=metrics.keys())
            self.writer.writeheader()
        self.writer.writerow(metrics)
        self.file.flush()

    def close(self):
        self.file.close()


def save_checkpoint(model, run_dir, filename="checkpoint.pt"):
    path = os.path.join(run_dir, filename)
    torch.save(model.state_dict(), path)
    return path

def append_global_trial_summary(summary: dict, base_dir):
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, "grid_summary.csv")

    file_exists = os.path.isfile(path)

    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=summary.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(summary)