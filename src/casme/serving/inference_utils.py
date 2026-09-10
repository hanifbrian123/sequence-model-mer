"""Shared deployment inference helpers with exact dataset preprocessing."""
import numpy as np
import torch

from casme.data.dataset import (estimate_apex_from_flow, prepare_seq_array,
                     resolve_apex_position)
from casme.training.engine import deterministic_tta_views


@torch.no_grad()
def predict_array(models, array, deploy, device, sample=None):
    """Average probabilities over checkpoints and deterministic eval views."""
    if not models:
        raise ValueError("at least one model checkpoint is required")
    if deploy.get("modality") == "two_stream":
        raise ValueError("predict_array expects a single-stream deployment")
    if sample is None:
        sample = {"apex_pos": -1}
        if (deploy.get("temporal_span") == "onset_apex"
                and deploy.get("modality") == "flow"):
            source = deploy.get("eval_apex_source", "annotation")
            if source == "fraction":
                sample["apex_pos"] = resolve_apex_position(
                    array, sample, deploy, train=False)
            else:
                # Annotation is unavailable for uploaded videos, so both an
                # explicit flow-energy recipe and an annotation-trained model
                # fall back to the fixed label-free estimator.
                sample["apex_pos"] = estimate_apex_from_flow(
                    array,
                    spatial_quantile=deploy.get("apex_energy_quantile", 0.95),
                    border_fraction=deploy.get("apex_border_fraction", 0.08),
                    smooth_radius=deploy.get("apex_smooth_radius", 0),
                    search_max_fraction=deploy.get(
                        "apex_search_max_fraction", 0.55))
    # ``sample`` is now fully resolved, including the automatic deployment
    # estimate.  Force canonical preprocessing to honor it so inference does
    # not estimate twice and an explicit --apex_index remains an override.
    preprocess_cfg = dict(deploy)
    preprocess_cfg["eval_apex_source"] = "annotation"
    views = deterministic_tta_views(deploy, deploy.get("tta", 1))
    probabilities = None
    for view in views:
        tensor = prepare_seq_array(
            array, sample, preprocess_cfg, train=False,
            view=view).unsqueeze(0).to(device)
        for model in models:
            with torch.autocast(device_type="cuda", enabled=(device == "cuda")):
                logits = model(tensor)
            current = torch.softmax(logits.float(), dim=1).cpu().numpy()[0]
            probabilities = current if probabilities is None else probabilities + current
    return probabilities / float(len(models) * len(views))
