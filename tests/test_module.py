#!/usr/bin/env python3
import random
import unittest
import torch


class BaseTest(unittest.TestCase):
    def setUp(self) -> None:
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def assertTensorAlmostEqual(
    self, actual: torch.Tensor, expected: torch.Tensor, delta: float = 0.0001
) -> None:
    """
    Args:

        self (): A unittest instance.
        actual (torch.Tensor): A tensor to compare with expected.
        expected (torch.Tensor): A tensor to compare with actual.
        delta (float, optional): The allowed difference between actual and expected.
            Default: 0.0001
    """
    self.assertEqual(actual.shape, expected.shape)
    self.assertEqual(actual.device == expected.device)
    self.assertEqual(actual.dtype == expected.dtype)
    self.assertAlmostEqual(
        torch.sum(torch.abs(actual - expected)).item(), 0.0, delta=delta
    )


def _create_test_faces(face_height = 512, face_width = 512):
    # Create unique colors for faces (6 colors)
    face_colors = [
        [0.0, 0.0, 0.0],
        [0.2, 0.2, 0.2],
        [0.4, 0.4, 0.4],
        [0.6, 0.6, 0.6],
        [0.8, 0.8, 0.8],
        [1.0, 1.0, 1.0],
    ]
    face_colors = torch.as_tensor(face_colors).view(6, 3, 1, 1)

    # Create and color faces (6 squares)
    faces = torch.ones([6, 3] + [face_height, face_width]) * face_colors
    return faces
