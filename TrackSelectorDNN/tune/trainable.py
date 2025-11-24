import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from ray.air import session
from ray.train import Checkpoint

from TrackSelectorDNN.models.track_classifier import TrackClassifier
from TrackSelectorDNN.data_manager.dataset_factory import get_dataset
from TrackSelectorDNN.configs.schema import load_config
from TrackSelectorDNN.tune.utils_logging import (
    create_run_dir,
    save_config,
    save_model_summary,
    CSVLogger
    #save_checkpoint
)

# ---------------------------
# Utility functions
# ---------------------------

def mirror_inputs(hit_features, track_features, idx_sym_hit_features, idx_sym_track_features):
    """
    Apply symmetric mirroring to specified hit and track features.

    Args:
        hit_features (torch.Tensor): Tensor of shape (batch_size, max_hits, hit_input_dim)
        track_features (torch.Tensor): Tensor of shape (batch_size, track_feat_dim)
        idx_sym_hit_features (list[int] or None): Indices of recHit features to mirror
        idx_sym_track_features (list[int] or None): Indices of track features to mirror

    Returns:
        tuple(torch.Tensor, torch.Tensor): Mirrored hit_features and track_features
    """
    
    hf = hit_features.clone() 
    tf = track_features.clone()

    if idx_sym_hit_features is not None:
        for idx in idx_sym_hit_features:
            hf[:,:,idx]*=-1
    if idx_sym_track_features is not None:
        for idx in idx_sym_track_features:
            tf[:,idx]*=-1
    return hf, tf
    

def train_one_epoch(model, loader, optimizer, device, idx_sym_hit_features, idx_sym_track_features, lambda_sym, w_true, w_fake):
    """
    Train the model for one epoch.

    Supports:
        - Weighted binary cross-entropy loss
        - Symmetry loss regularization
        - Batched training with DataLoader

    Args:
        model (nn.Module): TrackClassifier model
        loader (DataLoader): Training data loader
        optimizer (torch.optim.Optimizer): Optimizer for gradient updates
        device (str): "cuda" or "cpu"
        idx_sym_hit_features (list[int] or None): Indices of hit features to symmetrtize
        idx_sym_track_features (list[int] or None): Indices of track features to symmetrize
        lambda_sym (float or None): Weight for symmetry regularization
        w_true (torch.Tensor or None): Weight for positive labels
        w_fake (torch.Tensor or None): Weight for negative labels

    Returns:
        float: Average training loss for the epoch
    """
    
    model.train()
    total_loss = 0
    total_loss_sym = 0
    
    for batch in loader:
        hit_features = batch["hit_features"].to(device)
        track_features = batch["track_features"].to(device)
        mask = batch["mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        preds = model(hit_features, track_features, mask)
        
        if (w_true is not None)and (w_fake is not None):
            weight = torch.where(labels==1, w_true, w_fake)
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels, weight=weight)
        else:
            loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)

        if ((idx_sym_hit_features is not None) or (idx_sym_track_features is not None)) and lambda_sym is not None:
            hit_features_mirr, track_features_mirr = mirror_inputs(hit_features, track_features, idx_sym_hit_features, idx_sym_track_features)
            hit_features_mirr = hit_features_mirr.to(device)
            track_features_mirr = track_features_mirr.to(device)
            
            preds_mirr = model(hit_features_mirr, track_features_mirr, mask)
            loss_sym = lambda_sym * (torch.sigmoid(preds_mirr) - torch.sigmoid(preds)).pow(2).mean()
            loss += loss_sym
            total_loss_sym += loss_sym.item() * len(labels)
            
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * len(labels)
        
    return total_loss / len(loader.dataset), total_loss_sym / len(loader.dataset)


def validate(model, loader, device, idx_sym_hit_features, idx_sym_track_features, lambda_sym, w_true, w_fake):
    """
    Evaluate the model on a validation dataset.

    Args:
        model (nn.Module): TrackClassifier model
        loader (DataLoader): Validation data loader
        device (str): "cuda" or "cpu"
        idx_sym_hit_features (list[int] or None): Indices of hit features for symmetry
        idx_sym_track_features (list[int] or None): Indices of track features for symmetry
        lambda_sym (float or None): Weight for symmetry regularization
        w_true (torch.Tensor or None): Weight for positive labels
        w_fake (torch.Tensor or None): Weight for negative labels

    Returns:
        tuple(float, float): Validation loss and accuracy
    """
    
    model.eval()
    total_loss = 0
    total_loss_sym = 0
    correct = 0
    n = 0
    with torch.no_grad():
        for batch in loader:
            hit_features = batch["hit_features"].to(device)
            track_features = batch["track_features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["labels"].to(device)

            preds = model(hit_features, track_features, mask)
            if (w_true is not None)and (w_fake is not None):
                weight = torch.where(labels==1, w_true, w_fake)
                loss = nn.functional.binary_cross_entropy_with_logits(preds, labels, weight=weight)
            else:    
                loss = nn.functional.binary_cross_entropy_with_logits(preds, labels)

            if ((idx_sym_hit_features is not None) or (idx_sym_track_features is not None)) and lambda_sym is not None:
                hit_features_mirr, track_features_mirr = mirror_inputs(hit_features, track_features, idx_sym_hit_features, idx_sym_track_features)
                hit_features_mirr = hit_features_mirr.to(device)
                track_features_mirr = track_features_mirr.to(device)
                
                preds_mirr = model(hit_features_mirr, track_features_mirr, mask)
                loss_sym = lambda_sym * (torch.sigmoid(preds_mirr) - torch.sigmoid(preds)).pow(2).mean()
                total_loss_sym += loss_sym.item() * len(labels)

            
            total_loss += loss.item() * len(labels)
            preds_bin = (preds > 0).float()
            correct += (preds_bin == labels).sum().item()
            n += len(labels)
    return total_loss / n, correct / n, total_loss_sym/n


# ---------------------------
# Ray Tune Trainable
# ---------------------------
def trainable(config):
    """
    Ray Tune trainable function for TrackClassifier training.

    Handles:
        - Configuration parsing
        - Dataset and DataLoader creation
        - Model initialization and optimizer setup
        - Training loop with early stopping
        - Best model checkpointing and metric reporting to Ray

    Args:
        config (dict): Hyperparameter and training configuration dictionary
    """
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    patience = config["patience"]
    delta = config["delta"]
    base_checkpoint_directory = config["base_checkpoint_directory"]
    idx_sym_hit_features = config["idxSymRecHitFeatures"]
    idx_sym_track_features = config["idxSymRecoPixelTrackFeatures"]
    lambda_sym = config["lambda_sym"]
    w_fake = torch.tensor(config["w_fake"], device=device)
    w_true = torch.tensor(config["w_true"], device=device)
    
    # Trial directory for this run
    trial_name = None
    if session.get_session():
        try:
            trial_name = session.get_trial_id()
        except Exception:
            pass
            
    run_dir = create_run_dir(
        base_dir=base_checkpoint_directory,
        trial_name=trial_name
    )
    save_config(config, run_dir)

    # Dataset
    train_ds, collate_fn = get_dataset(config, dataset_role="train_path")
    val_ds, _ = get_dataset(config, dataset_role="val_path")
    
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"],
                              shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"], collate_fn=collate_fn)

    # Model
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

    optimizer = optim.Adam(model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"])

    logger = CSVLogger(run_dir)
    n_epochs = config["epochs"]

    # Initialize best metrics safely
    best_val_loss = float("inf")
    best_metrics = {"val_loss": float("inf"), "val_acc": 0.0, "epoch": 0}
    count_stopping = 0
    # Training loop
    for epoch in range(n_epochs):
        if(count_stopping==patience):
            break
        train_loss, train_loss_sym = train_one_epoch(model, train_loader, optimizer, device, idx_sym_hit_features, idx_sym_track_features, lambda_sym, w_true, w_fake)
        val_loss, val_acc, val_loss_sym = validate(model, val_loader, device, idx_sym_hit_features, idx_sym_track_features, lambda_sym, w_true, w_fake)

        metrics = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_loss_sym": train_loss_sym,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_loss_sym": val_loss_sym,
        }

        print(f"[Epoch {epoch+1}/{n_epochs}] "
              f"train_loss={train_loss:.4f}, val_loss={val_loss:.4f}, val_acc={val_acc:.4f}")
        logger.log(metrics)

        # Save best checkpoint only
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_metrics = metrics

            # Save lightweight checkpoint directory
            torch.save(model.state_dict(), os.path.join(run_dir, "best_model.pt"))

            with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
                json.dump(best_metrics, f)

            session.report(
                metrics,
                checkpoint=Checkpoint.from_directory(run_dir)
            )
            count_stopping = 0
        if val_loss + delta > best_val_loss :
            count_stopping+=1

    # Final report once training ends
    print("[DEBUG] Final report to Ray:", run_dir)
    with open(os.path.join(run_dir, "best_metrics.json"), "w") as f:
        json.dump(best_metrics, f)

    session.report(
        best_metrics,
        checkpoint=Checkpoint.from_directory(run_dir)
    )
    

if __name__ == "__main__":
    cfg = load_config("base.yaml")
    print(cfg)
    trainable(cfg.dict())
