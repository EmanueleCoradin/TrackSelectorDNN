from ray import tune
from TrackSelectorDNN.configs.schema import load_config
from TrackSelectorDNN.tune.trainable import trainable

base_cfg = load_config("base.yaml").model_dump()

search_space = {
    **base_cfg,  # include all validated defaults
    "lr": tune.loguniform(1e-4, 1e-2),
    "latent_dim": tune.choice([8, 16, 32]),
    "pooling_type": tune.choice(["sum", "mean", "softmax"]),
    "netA_hidden_layers": tune.choice([1, 2, 3]),
    "netB_hidden_layers": tune.choice([1, 2, 3]),
}

tuner = tune.Tuner(
    trainable,
    param_space=search_space,
    tune_config=tune.TuneConfig(metric="val_loss", mode="min", num_samples=20),
)
tuner.fit()
