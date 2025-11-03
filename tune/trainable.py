import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from TrackSelectorDNN.models.track_classifier import TrackClassifier
from TrackSelectorDNN.data_manager.dataset_factory import get_dataset

from ray import tune
from ray.air import session

from TrackSelectorDNN.configs.schema import load_config
from TrackSelectorDNN.tune.utils_logging import create_run_dir, save_config, save_model_summary, CSVLogger, save_checkpoint

# ---------------------------
# Utility: simple train/val loop
# ---------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in loader:
        hit_features = batch["hit_features"].to(device)
        track_features = batch["track_features"].to(device)
        batch_indices = batch["batch_indices"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        preds = model(hit_features, track_features, batch_indices)
        loss = criterion(preds, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
    return total_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            hit_features = batch["hit_features"].to(device)
            track_features = batch["track_features"].to(device)
            batch_indices = batch["batch_indices"].to(device)
            labels = batch["labels"].to(device)

            preds = model(hit_features, track_features, batch_indices)
            loss = criterion(preds, labels)
            total_loss += loss.item() * len(labels)

            preds_bin = (preds > 0.5).float()
            correct += (preds_bin == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n


# ---------------------------
# Main trainable (Ray-compatible)
# ---------------------------
def trainable(config, checkpoint_dir=None):
    """
    Trainable function compatible with Ray Tune.
    If run standalone, behaves like a normal training script.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # --- Setup logs ---
    trial_name = None
    if session.get_session():
        try:
            trial_name = tune.get_tiral_name()
        except Exception:
            pass
            
    run_dir = create_run_dir(base_dir="/eos/user/e/ecoradin/GitHub/TrackSelectorDNN/runs", trial_name=trial_name)
    save_config(config, run_dir)
    
    # --- Load dataset ---
    dataset, collate_fn = get_dataset(config)
    val_fraction = config['val_fraction']
    val_len   = int(len(dataset) * val_fraction)
    train_len = len(dataset) - val_len
    train_ds, val_ds = random_split(dataset, [train_len, val_len])

    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=collate_fn)

    # --- Build model ---
    model = TrackClassifier(
        hit_input_dim=config["hit_input_dim"],
        track_feat_dim=config["track_feat_dim"],
        latent_dim=config["latent_dim"],
        pooling_type=config["pooling_type"],
        
        netA_hidden_dim=config["netA_hidden_dim"],
        netA_hidden_layers=config["netA_hidden_layers"],
        netA_batchnorm=config["netA_batchnorm"],
        netA_activation=config["netA_activation"],
        
        netB_hidden_dim=config["netB_hidden_dim"],
        netB_hidden_layers=config["netB_hidden_layers"],
        netB_batchnorm=config["netB_batchnorm"],
        netB_activation=config["netB_activation"],
    ).to(device)

    save_model_summary(model, run_dir)

    # --- Optimizer / Loss ---
    optimizer = optim.Adam(model.parameters(), lr=config["lr"])
    criterion = nn.BCELoss()

    # --- Training loop ---
    logger = CSVLogger(run_dir)
    best_val_loss = float("inf")
    best_path = None
    n_epochs = config["epochs"]
    for epoch in range(n_epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        print(f"[Epoch {epoch+1}/{n_epochs}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        logger.log(metrics)

        # Save Best Model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_path = save_checkpoint(model, run_dir, filename="best_model.pt")
            
        # Ray Tune reporting
        if "tune" in globals() and session.get_session():
            tune.report(metrics)

    # --- Save final checkpoint ---
    save_checkpoint(model, run_dir, filename="final_model.pt")
    logger.close()

    return {"val_loss": val_loss}
    
if __name__ == "__main__":
    # Load and validate the configuration
    cfg = load_config("configs/base.yaml")
    print(cfg)  # validated dataclass-like object

    # Convert to dict for trainable
    trainable(cfg.dict())
    
