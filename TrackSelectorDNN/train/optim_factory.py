"""
Module for constructing optimizers and learning-rate schedulers.
"""

import torch
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, OneCycleLR, ReduceLROnPlateau


def build_optimizer(model, config):
    """
    Construct and return the optimizer specified in the configuration.

    Parameters
    ----------
    model : torch.nn.Module
        Model whose parameters will be optimized.

    config : Config
        Global configuration object containing `training.optimizer` settings.
        Expected fields:
            - optimizer.name : {"adam", "adamw"}
            - optimizer.lr : float
            - optimizer.weight_decay : float

    Returns
    -------
    torch.optim.Optimizer
        A fully initialized optimizer instance.

    Raises
    ------
    ValueError
        If an unsupported optimizer name is provided.
    """
    optim_cfg = config.training.optimizer
    name = optim_cfg.name.lower()

    lr = optim_cfg.lr
    wd = optim_cfg.weight_decay

    if name == "adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "adamw":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def build_scheduler(optimizer, config, total_steps=None):
    """
    Construct and return a learning-rate scheduler as defined in the configuration.

    Supported schedulers:
        - "none": no scheduler returned.
        - "plateau": ReduceLROnPlateau (epoch-level)
        - "cosine": CosineAnnealingLR (epoch-level)
        - "onecycle": OneCycleLR (batch-level)

    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        The optimizer whose learning rate will be modified.

    config : Config
        Global configuration object containing `training.scheduler` settings.

    total_steps : int, optional
        Number of optimizer steps, required if the scheduler
        is OneCycleLR (batch-level scheduling).

    Returns
    -------
    tuple or None
        If no scheduler is configured:
            None

        Otherwise:
            (scheduler, scheduler_type)

            scheduler : torch.optim.lr_scheduler._LRScheduler
                Initialized scheduler instance.

            scheduler_type : str
                "epoch" → call scheduler.step() once per epoch  
                "batch" → call scheduler.step() once per training batch

    Raises
    ------
    ValueError
        If an unknown scheduler name is given, or if OneCycleLR is selected
        without providing total_steps.
    """
    sched_cfg = config.training.scheduler
    name = sched_cfg.name.lower()
    scheduler, scheduler_type = (None, None)

    if name == "none":
        return None, None

    if name == "plateau":
        scheduler = ReduceLROnPlateau(
            optimizer,
            factor=sched_cfg.factor,
            patience=sched_cfg.patience,
            min_lr=sched_cfg.min_lr
        )
        scheduler_type = "epoch"

    elif name == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=sched_cfg.T_max,
            eta_min=sched_cfg.eta_min
        )
        scheduler_type = "epoch"

    elif name == "onecycle":
        if total_steps is None:
            raise ValueError("OneCycleLR requires total_steps")

        scheduler = OneCycleLR(
            optimizer,
            max_lr=sched_cfg.max_lr,
            total_steps=total_steps,
            pct_start=sched_cfg.pct_start,
            anneal_strategy=sched_cfg.anneal_strategy,
            div_factor=sched_cfg.div_factor,
            final_div_factor=sched_cfg.final_div_factor,
            three_phase=sched_cfg.three_phase
        )
        scheduler_type = "batch"

    else:
        raise ValueError(f"Unknown scheduler: {name}")

    return scheduler, scheduler_type
