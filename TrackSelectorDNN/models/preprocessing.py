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

    def _match_shape(self, p, x):
        if p.numel() == 0:
            return p
        while p.dim() < x.dim():
            p = p.unsqueeze(0)
        return p.expand_as(x)
        
    def forward(self, x):
        
        if self.do_log.numel() > 0:
            log_mask = self._match_shape(self.do_log, x)
            x = torch.where(log_mask, torch.log10(x + 1e-8), x)
            
        if self.clip_min.numel() > 0 or self.clip_max.numel() > 0:
            clip_min = self._match_shape(self.clip_min, x) if self.clip_min.numel() > 0 else None
            clip_max = self._match_shape(self.clip_max, x) if self.clip_max.numel() > 0 else None
            x.clamp_(min=clip_min, max=clip_max)
        
        if self.mean.numel() > 0 and self.std.numel() > 0:
            mean = self._match_shape(self.mean, x)
            std  = self._match_shape(self.std, x)
            x.sub_(mean)
            x.div_(std + 1e-8)
            
        return x
