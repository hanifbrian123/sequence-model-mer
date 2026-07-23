import copy
import os
import sys
import unittest
from unittest import mock

import numpy as np
import torch


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import dataset  # noqa: E402
import engine  # noqa: E402
import inference_utils  # noqa: E402


class TinyModel(torch.nn.Module):
    def forward(self, x):
        if isinstance(x, (list, tuple)):
            x = x[0]
        score = x.float().mean(dim=(1, 2, 3, 4))
        return torch.stack((score, -score), dim=1)


def base_cfg(**updates):
    cfg = {
        "T": 4,
        "img_size": 2,
        "batch_size": 1,
        "input_mode": "diff",
        "modality": "rgb",
        "temporal_jitter": True,
        "hflip": True,
        "color_jitter": 0.5,
        "random_erase": 1.0,
        "amp": False,
    }
    cfg.update(updates)
    return cfg


def synthetic_inputs():
    # Each time step and spatial location is unique, making crop/time mistakes
    # observable without touching the licensed dataset.
    arr = np.arange(20 * 4 * 4 * 3, dtype=np.uint16).reshape(20, 4, 4, 3)
    arr = (arr % 256).astype(np.uint8)
    samples = [{"key": "synthetic", "label": 0, "apex_pos": -1}]
    return samples, {"synthetic": arr}


class EvalPipelineTests(unittest.TestCase):
    def test_tta_keeps_eval_mode_and_applies_deterministic_shift(self):
        samples, arrays = synthetic_inputs()
        cfg = base_cfg(eval_rand_start=True, eval_start_fraction=0.25)
        original = dataset.sample_indices
        calls = []

        def spy(*args, **kwargs):
            calls.append((args, copy.deepcopy(kwargs)))
            return original(*args, **kwargs)

        with mock.patch.object(dataset, "sample_indices", side_effect=spy), \
                mock.patch.object(dataset, "random_erase",
                                  side_effect=AssertionError("eval called random erase")):
            engine.predict(TinyModel(), samples, arrays, cfg, "cpu", tta=5)

        self.assertEqual(len(calls), 5)
        self.assertTrue(all(args[2] is False for args, _ in calls))
        self.assertTrue(all(kwargs["lo_idx"] == 5 for _, kwargs in calls))

    def test_tta_is_reproducible_and_does_not_consume_numpy_rng(self):
        samples, arrays = synthetic_inputs()
        cfg = base_cfg(eval_rand_start=True, eval_start_fraction=0.25)
        np.random.seed(1234)
        before = copy.deepcopy(np.random.get_state())
        p1 = engine.predict(TinyModel(), samples, arrays, cfg, "cpu", tta=5)
        after_first = copy.deepcopy(np.random.get_state())
        p2 = engine.predict(TinyModel(), samples, arrays, cfg, "cpu", tta=5)
        after_second = np.random.get_state()

        np.testing.assert_allclose(p1, p2, rtol=0, atol=0)
        for a, b in zip(before, after_first):
            np.testing.assert_array_equal(a, b)
        for a, b in zip(after_first, after_second):
            np.testing.assert_array_equal(a, b)

    def test_center_view_matches_manual_inference_preprocessing(self):
        samples, arrays = synthetic_inputs()
        cfg = base_cfg(random_erase=0.0, color_jitter=0.0)
        ds = dataset.SeqDataset(samples, arrays, cfg, train=False,
                                view={"crop": "center", "flip": False,
                                      "temporal_phase": 0.5})
        actual, label = ds[0]

        arr = arrays["synthetic"]
        idx = dataset.sample_indices(len(arr), cfg["T"], train=False,
                                     jitter=cfg["temporal_jitter"], phase=0.5)
        clip = arr[idx]
        top, left = dataset.pick_crop(clip.shape[1], clip.shape[2],
                                      cfg["img_size"], train=False, mode="center")
        expected = dataset._to_cthw(dataset.build_rgb(
            clip, top, left, cfg["img_size"], False, None, cfg["input_mode"]))

        self.assertEqual(label, 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_explicit_tta_views_are_honored(self):
        cfg = base_cfg(tta_views=[
            {"crop": "center", "flip": False, "temporal_phase": 0.25},
            {"crop": "center", "flip": True, "temporal_phase": 0.75},
        ])
        views = engine.deterministic_tta_views(cfg, tta=5)
        self.assertEqual(views, cfg["tta_views"])

    def test_deployment_preprocessing_matches_evaluator_exactly(self):
        samples, arrays = synthetic_inputs()
        cfg = base_cfg(tta=5, eval_rand_start=True,
                       eval_start_fraction=0.25)
        model = TinyModel()
        evaluator = engine.predict(
            model, samples, arrays, cfg, "cpu", tta=cfg["tta"])
        deployment = inference_utils.predict_array(
            [model], arrays["synthetic"], cfg, "cpu",
            sample={"apex_pos": -1})
        np.testing.assert_allclose(deployment, evaluator[0], rtol=0, atol=0)

    def test_flow_translation_compensation_removes_global_motion(self):
        flow = np.zeros((2, 5, 5, 2), dtype=np.float32)
        flow[..., 0] = 1.25
        flow[..., 1] = -0.75
        flow[:, 2, 2, 0] += 0.5
        output = dataset.build_flow(
            flow, 0, 0, 5, False, flow_clip=3.0,
            compensation="translation")
        np.testing.assert_allclose(np.median(output[..., 0], axis=(1, 2)), 0.0,
                                   rtol=0, atol=1e-7)
        np.testing.assert_allclose(np.median(output[..., 1], axis=(1, 2)), 0.0,
                                   rtol=0, atol=1e-7)

    def test_soft_face_roi_suppresses_flow_corners(self):
        flow = np.ones((2, 9, 9, 2), dtype=np.float32)
        output = dataset.build_flow(
            flow, 0, 0, 9, False, flow_clip=3.0, roi="face_ellipse")
        self.assertAlmostEqual(float(output[0, 0, 0, 0]), 0.0, places=6)
        self.assertGreater(float(output[0, 4, 4, 0]), 0.3)

    def test_flow_energy_apex_estimator_finds_smoothed_peak(self):
        flow = np.zeros((9, 8, 8, 2), dtype=np.float32)
        flow[4, 2:6, 2:6, 0] = 2.0
        flow[3, 2:6, 2:6, 0] = 0.8
        flow[5, 2:6, 2:6, 0] = 0.8
        self.assertEqual(dataset.estimate_apex_from_flow(flow), 4)

    def test_flow_energy_apex_estimator_respects_search_cap(self):
        flow = np.zeros((11, 8, 8, 2), dtype=np.float32)
        flow[4, ..., 0] = 1.0
        flow[9, ..., 0] = 3.0
        self.assertEqual(dataset.estimate_apex_from_flow(
            flow, smooth_radius=0, search_max_fraction=0.6), 4)

    def test_deployment_auto_apex_matches_explicit_estimate(self):
        flow = np.zeros((9, 4, 4, 2), dtype=np.float32)
        flow[4, ..., 0] = 2.0
        cfg = base_cfg(modality="flow", temporal_span="onset_apex",
                       flow_clip=3.0, color_jitter=0.0, random_erase=0.0)
        model = TinyModel()
        apex = dataset.estimate_apex_from_flow(flow)
        automatic = inference_utils.predict_array([model], flow, cfg, "cpu")
        explicit = inference_utils.predict_array(
            [model], flow, cfg, "cpu", sample={"apex_pos": apex})
        np.testing.assert_allclose(automatic, explicit, rtol=0, atol=0)

    def test_tta_apex_fraction_overrides_annotation(self):
        flow = np.zeros((11, 4, 4, 2), dtype=np.float32)
        cfg = base_cfg(modality="flow", temporal_span="onset_apex",
                       flow_clip=3.0, color_jitter=0.0, random_erase=0.0)
        with mock.patch.object(dataset, "sample_indices",
                               wraps=dataset.sample_indices) as spy:
            dataset.prepare_seq_array(
                flow, {"apex_pos": 9}, cfg, train=False,
                view={"apex_fraction": 0.5})
        self.assertEqual(spy.call_args.kwargs["hi_idx"], 5)


if __name__ == "__main__":
    unittest.main()
