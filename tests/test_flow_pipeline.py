import os
import sys
import unittest

import cv2
import numpy as np


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

import flow_pipeline  # noqa: E402


class FlowPipelineTests(unittest.TestCase):
    def test_ecc_stabilization_reduces_translation_error(self):
        rng = np.random.default_rng(7)
        reference = rng.integers(0, 256, size=(64, 64), dtype=np.uint8)
        reference = cv2.GaussianBlur(reference, (5, 5), 0)
        transform = np.array([[1.0, 0.0, 3.0], [0.0, 1.0, -2.0]],
                             dtype=np.float32)
        shifted = cv2.warpAffine(
            reference, transform, (64, 64), borderMode=cv2.BORDER_REFLECT_101)
        aligned, success = flow_pipeline.stabilize_to_reference(
            reference, shifted, motion="translation")
        before = np.mean((shifted.astype(np.float32) - reference) ** 2)
        after = np.mean((aligned.astype(np.float32) - reference) ** 2)
        self.assertTrue(success)
        self.assertLess(after, before * 0.25)

    def test_hq_tvl1_parameters_are_explicit(self):
        optical_flow = flow_pipeline.make_tvl1("hq")
        self.assertEqual(optical_flow.getScalesNumber(), 7)
        self.assertEqual(optical_flow.getWarpingsNumber(), 10)
        self.assertEqual(optical_flow.getOuterIterations(), 15)
        self.assertEqual(optical_flow.getInnerIterations(), 40)

    def test_onset_flow_shape_and_identity_frame(self):
        reference = np.zeros((32, 32), dtype=np.uint8)
        cv2.circle(reference, (16, 16), 6, 200, -1)
        shifted = np.roll(reference, 1, axis=1)
        flow, failures = flow_pipeline.onset_flow_from_grays(
            [reference, shifted], preset="default", stabilize="none")
        self.assertEqual(flow.shape, (2, 32, 32, 2))
        self.assertEqual(failures, 0)
        self.assertLess(float(np.abs(flow[0]).max()), 1e-5)


if __name__ == "__main__":
    unittest.main()
