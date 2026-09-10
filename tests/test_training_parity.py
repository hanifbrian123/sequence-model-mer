import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from casme.training import engine  # noqa: E402
from casme.training import train_final  # noqa: E402


class TinyClassifier(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = torch.nn.AdaptiveAvgPool3d(1)
        self.head = torch.nn.Linear(3, 2)

    def forward(self, x):
        return self.head(self.pool(x).flatten(1))


def tiny_training_inputs():
    samples = []
    arrays = {}
    for index in range(6):
        key = f"sample_{index}"
        value = 20 + index * 35
        arrays[key] = np.full((6, 4, 4, 3), value, dtype=np.uint8)
        samples.append({"key": key, "label": int(index >= 3), "apex_pos": -1})
    return samples, arrays


def tiny_cfg(**updates):
    cfg = {
        "backbone": "tiny",
        "modality": "rgb",
        "input_mode": "rgb",
        "T": 4,
        "img_size": 4,
        "resize_to": None,
        "batch_size": 3,
        "epochs": 3,
        "lr": 0.02,
        "weight_decay": 0.0,
        "class_weighting": False,
        "loss": "ce",
        "label_smoothing": 0.0,
        "temporal_jitter": False,
        "hflip": False,
        "color_jitter": 0.0,
        "random_erase": 0.0,
        "amp": False,
        "eval_last_k": 2,
        "seed": 17,
    }
    cfg.update(updates)
    return cfg


class TrainingParityTests(unittest.TestCase):
    def test_configured_focal_loss_is_used_by_shared_builder(self):
        samples, _ = tiny_training_inputs()
        criterion = engine.build_criterion(
            samples, tiny_cfg(loss="focal", focal_gamma=1.5), 2, "cpu")
        self.assertIsInstance(criterion, engine.FocalLoss)
        self.assertEqual(criterion.gamma, 1.5)

    def test_train_final_collects_configured_tail_checkpoints(self):
        samples, arrays = tiny_training_inputs()
        cfg = tiny_cfg(eval_last_k=2, epochs=3)
        with mock.patch.object(train_final, "build_model",
                               side_effect=lambda *_: TinyClassifier()):
            snapshots = train_final.train_all(
                17, samples, arrays, cfg, 2, "cpu", lambda _: None)

        self.assertEqual([s["epoch"] for s in snapshots], [1, 2])
        self.assertTrue(all(s["weights"] == "raw" for s in snapshots))
        for snapshot in snapshots:
            self.assertTrue(all(t.device.type == "cpu"
                                for t in snapshot["state_dict"].values()))
        self.assertFalse(torch.equal(snapshots[0]["state_dict"]["head.weight"],
                                     snapshots[1]["state_dict"]["head.weight"]))

    def test_ema_snapshot_does_not_mutate_live_model(self):
        model = TinyClassifier()
        ema = engine.EMA(model, decay=0.5)
        with torch.no_grad():
            for parameter in model.parameters():
                parameter.add_(1.0)
        live_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        state = train_final._snapshot_state(model, ema=ema)

        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, live_before[key])
        self.assertFalse(torch.equal(state["head.weight"], live_before["head.weight"]))

    def test_locked_audit_skips_per_epoch_validation(self):
        samples, arrays = tiny_training_inputs()
        train_samples, audit_samples = samples[:4], samples[4:]
        cfg = tiny_cfg(monitor_val=False, epochs=3, eval_last_k=2)
        original_evaluate = engine.evaluate_val
        with mock.patch.object(engine, "build_model",
                               side_effect=lambda *_: TinyClassifier()), \
                mock.patch.object(engine, "evaluate_val",
                                  wraps=original_evaluate) as evaluate:
            probabilities, history = engine.train_fold(
                train_samples, audit_samples, arrays, cfg, 2, "cpu",
                lambda _: None)

        self.assertEqual(evaluate.call_count, 2)
        self.assertEqual(probabilities.shape, (2, 2))
        self.assertTrue(all(row["val_loss"] is None for row in history))

    def test_fold_snapshot_callback_receives_each_collected_tail(self):
        samples, arrays = tiny_training_inputs()
        received = []
        with mock.patch.object(engine, "build_model",
                               side_effect=lambda *_: TinyClassifier()):
            engine.train_fold(
                samples[:4], samples[4:], arrays,
                tiny_cfg(epochs=3, eval_last_k=2), 2, "cpu", lambda _: None,
                snapshot_fn=lambda epoch, state: received.append((epoch, state)))
        self.assertEqual([epoch for epoch, _ in received], [1, 2])
        self.assertTrue(all(all(value.device.type == "cpu"
                                for value in state.values())
                            for _, state in received))


if __name__ == "__main__":
    unittest.main()
