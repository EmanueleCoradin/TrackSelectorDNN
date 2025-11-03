import yaml
from models.registry import get_activation

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)