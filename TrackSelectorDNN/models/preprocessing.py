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

        def _buf(x):
            return x if x is not None else torch.empty(0)

        self.register_buffer("mean",     _buf(mean))
        self.register_buffer("std",      _buf(std))
        self.register_buffer("clip_min", _buf(torch.nan_to_num(clip_min, nan=-float("inf"))))
        self.register_buffer("clip_max", _buf(torch.nan_to_num(clip_max, nan=float("inf"))))
        self.register_buffer("do_log",   _buf(do_log))
        
    def forward(self, x):
        """
        x: (B, F) or (B, T, F)
        """

        # Broadcast do_log to last dimension
        if self.do_log.numel() > 0:
            # Add dims until shapes match
            log_mask = self.do_log
            while log_mask.dim() < x.dim():
                log_mask = log_mask.unsqueeze(0)

            x = torch.where(log_mask, torch.log10(x + 1e-8), x)

        # Clipping
        if self.clip_min.numel() > 0:
            clip_min = self.clip_min
            while clip_min.dim() < x.dim():
                clip_min = clip_min.unsqueeze(0)
            x = torch.maximum(x, clip_min)

        if self.clip_max.numel() > 0:
            clip_max = self.clip_max
            while clip_max.dim() < x.dim():
                clip_max = clip_max.unsqueeze(0)
            x = torch.minimum(x, clip_max)

        # Normalization
        if self.mean.numel() > 0 and self.std.numel() > 0:
            mean = self.mean
            std  = self.std
            while mean.dim() < x.dim():
                mean = mean.unsqueeze(0)
                std  = std.unsqueeze(0)

            x = (x - mean) / (std + 1e-8)

        return x
