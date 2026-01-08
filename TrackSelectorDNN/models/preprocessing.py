"""
Module for feature preprocessing: log transform, clipping, normalization.
"""

import torch
import torch.nn as nn

class FeaturePreprocessing(nn.Module):
    """
    Apply (optional) log10 transform, clipping, and normalization to the
    *last feature dimension*. Works with both:
        - track features:          (B, F)
        - per-hit features:        (B, T, F)
    """

    def __init__(self, mean=None, std=None, clip_min=None, clip_max=None, do_log=None):
        """
        Args:
            mean:     tensor (..., F) or None
            std:      tensor (..., F) or None
            clip_min: tensor (..., F) or None
            clip_max: tensor (..., F) or None
            do_log:   boolean tensor (..., F) or None
        """
        super().__init__()

        # Register buffers (saved in state_dict & ONNX-safe)
        for name, val in [
            ("mean", mean),
            ("std", std),
            ("clip_min", torch.nan_to_num(clip_min, nan=-float("inf"))),
            ("clip_max", torch.nan_to_num(clip_max, nan=float("inf"))),
            ("do_log", do_log),
        ]:
            if val is not None:
                self.register_buffer(name, val)
            else:
                setattr(self, name, None)

    def forward(self, x):
        """
        x: (B, F) or (B, T, F)
        """

        # Broadcast do_log to last dimension
        if self.do_log is not None:
            # Add dims until shapes match
            log_mask = self.do_log
            while log_mask.dim() < x.dim():
                log_mask = log_mask.unsqueeze(0)

            x = torch.where(log_mask, torch.log10(x + 1e-8), x)

        # Clipping
        if self.clip_min is not None:
            clip_min = self.clip_min
            while clip_min.dim() < x.dim():
                clip_min = clip_min.unsqueeze(0)
            x = torch.maximum(x, clip_min)

        if self.clip_max is not None:
            clip_max = self.clip_max
            while clip_max.dim() < x.dim():
                clip_max = clip_max.unsqueeze(0)
            x = torch.minimum(x, clip_max)

        # Normalization
        if self.mean is not None and self.std is not None:
            mean = self.mean
            std  = self.std
            while mean.dim() < x.dim():
                mean = mean.unsqueeze(0)
                std  = std.unsqueeze(0)

            x = (x - mean) / (std + 1e-8)

        return x
