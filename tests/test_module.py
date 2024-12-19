#!/usr/bin/env python3
import math
import random
import unittest

import numpy as np

import torch
from pytorch360convert.pytorch360convert import (
    c2e,
    coor2uv,
    e2c,
    equirect_facetype,
    equirect_uvgrid,
    grid_sample_wrap,
    rotation_matrix,
    uv2coor,
    uv2unitxyz,
    xyz2uv,
    xyzcube,
)


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
    self.assertEqual(actual.device, expected.device)
    self.assertEqual(actual.dtype, expected.dtype)
    self.assertAlmostEqual(
        torch.sum(torch.abs(actual - expected)).item(), 0.0, delta=delta
    )


def _create_test_faces(face_height: int = 512, face_width: int = 512) -> torch.Tensor:
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


class TestFunctionsBaseTest(unittest.TestCase):
    def setUp(self) -> None:
        seed = 1234
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True

    def test_rotation_matrix(self) -> None:
        # Test identity rotation (0 radians around any axis)
        axis = torch.tensor([1.0, 0.0, 0.0])
        angle = torch.tensor(0.0)
        result = rotation_matrix(angle, axis)
        expected = torch.eye(3)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

        # Test 90-degree rotation around x-axis
        angle = torch.tensor(math.pi / 2)
        result = rotation_matrix(angle, axis)
        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

        # Test rotation matrix properties
        # Should be orthogonal (R * R.T = I)
        result_t = result.t()
        identity = torch.mm(result, result_t)
        torch.testing.assert_close(identity, torch.eye(3), rtol=1e-6, atol=1e-6)

    def test_xyzcube(self) -> None:
        face_w = 4
        result = xyzcube(face_w)

        # Check shape
        self.assertEqual(result.shape, (face_w, face_w * 6, 3))

        # Check that coordinates are normalized (-0.5 to 0.5)
        self.assertTrue(torch.all(result >= -0.5))
        self.assertTrue(torch.all(result <= 0.5))

        # Test front face center point (adjusting for coordinate system)
        center_idx = face_w // 2
        front_center = result[center_idx, center_idx]
        expected_front = torch.tensor([0.0, 0.0, 0.5])
        torch.testing.assert_close(front_center, expected_front, rtol=0.17, atol=0.17)

    def test_equirect_uvgrid(self) -> None:
        h, w = 8, 16
        result = equirect_uvgrid(h, w)

        # Check shape
        self.assertEqual(result.shape, (h, w, 2))

        # Check ranges
        u = result[..., 0]
        v = result[..., 1]
        self.assertTrue(torch.all(u >= -torch.pi))
        self.assertTrue(torch.all(u <= torch.pi))
        self.assertTrue(torch.all(v >= -torch.pi / 2))
        self.assertTrue(torch.all(v <= torch.pi / 2))

        # Check center point
        center_h, center_w = h // 2, w // 2
        center_point = result[center_h, center_w]
        expected_center = torch.tensor([0.0, 0.0])
        torch.testing.assert_close(
            center_point, expected_center, rtol=0.225, atol=0.225
        )

    def test_equirect_facetype(self) -> None:
        h, w = 8, 16
        result = equirect_facetype(h, w)

        # Check shape
        self.assertEqual(result.shape, (h, w))

        # Check face type range (0-5 for 6 faces)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 5))

        # Check dtype
        self.assertEqual(result.dtype, torch.int32)

    def test_xyz2uv_and_uv2unitxyz(self) -> None:
        # Create test points
        xyz = torch.tensor(
            [
                [1.0, 0.0, 0.0],  # right
                [0.0, 1.0, 0.0],  # up
                [0.0, 0.0, 1.0],  # front
            ]
        )

        # Convert xyz to uv
        uv = xyz2uv(xyz)

        # Convert back to xyz
        xyz_reconstructed = uv2unitxyz(uv)

        # Normalize input xyz for comparison
        xyz_normalized = torch.nn.functional.normalize(xyz, dim=-1)

        # Verify reconstruction
        torch.testing.assert_close(
            xyz_normalized, xyz_reconstructed, rtol=1e-6, atol=1e-6
        )

    def test_uv2coor_and_coor2uv(self) -> None:
        h, w = 8, 16
        # Create test UV coordinates
        test_uv = torch.tensor(
            [
                [0.0, 0.0],  # center
                [torch.pi / 2, 0.0],  # right quadrant
                [-torch.pi / 2, 0.0],  # left quadrant
            ]
        )

        # Convert UV to image coordinates
        coor = uv2coor(test_uv, h, w)

        # Convert back to UV
        uv_reconstructed = coor2uv(coor, h, w)

        # Verify reconstruction
        torch.testing.assert_close(test_uv, uv_reconstructed, rtol=1e-5, atol=1e-5)

    def test_grid_sample_wrap(self) -> None:
        # Create test image
        h, w = 4, 8
        channels = 3
        image = torch.arange(h * w * channels, dtype=torch.float32)
        image = image.reshape(h, w, channels)

        # Test basic sampling
        coor_x = torch.tensor([[1.5, 2.5], [3.5, 4.5]])
        coor_y = torch.tensor([[1.5, 1.5], [2.5, 2.5]])

        # Test both interpolation modes
        result_bilinear = grid_sample_wrap(image, coor_x, coor_y, mode="bilinear")
        result_nearest = grid_sample_wrap(image, coor_x, coor_y, mode="nearest")

        # Check shapes
        self.assertEqual(result_bilinear.shape, (2, 2, channels))
        self.assertEqual(result_nearest.shape, (2, 2, channels))

        # Test horizontal wrapping
        wrap_x = torch.tensor([[w - 1.5, 0.5]])
        wrap_y = torch.tensor([[1.5, 1.5]])
        result_wrap = grid_sample_wrap(image, wrap_x, wrap_y, mode="bilinear")

        # Check that wrapped coordinates produce similar values
        # We use a larger tolerance here due to interpolation differences
        torch.testing.assert_close(
            result_wrap[0, 0],
            result_wrap[0, 1],
            rtol=0.5,
            atol=0.5,
        )

    def test_c2e_then_e2c(self) -> None:
        face_width = 512
        test_faces = _create_test_faces(face_width, face_width)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])
        assertTensorAlmostEqual(self, cubic_img, test_faces)

    def test_c2e_then_e2c_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        face_width = 512
        test_faces = _create_test_faces(face_width, face_width)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])
        self.assertTrue(cubic_img.is_cuda)
        assertTensorAlmostEqual(self, cubic_img, test_faces)

    def test_c2e_stack_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertTrue(equi_img.requires_grad)

    def test_e2c_stack_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4], requires_grad=True)
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])
        self.assertTrue(cubic_img.requires_grad)

    def test_c2e_then_e2c_stack_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])
        self.assertTrue(cubic_img.requires_grad)

    def test_c2e_list_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        test_faces = [test_faces[i] for i in range(test_faces.shape[0])]
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertTrue(equi_img.requires_grad)

    def test_e2c_list_grad(self) -> None:
        face_width = 512
        equi_img = torch.ones([3, face_width * 2, face_width * 4], requires_grad=True)
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="list"
        )
        for i in range(6):
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])
        for i in range(6):
            self.assertTrue(cubic_img[i].requires_grad)

    def test_c2e_then_e2c_list_grad(self) -> None:
        face_width = 512
        test_faces_tensors = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        test_faces = {k: test_faces_tensors[i] for i, k in zip(range(6), dict_keys)}
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        for i in range(6):
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])
        for i in range(6):
            self.assertTrue(cubic_img[i].requires_grad)

    def test_c2e_dict_grad(self) -> None:
        dict_keys =  ["Front", "Right", "Back", "Left", "Up", "Down"]
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        test_faces = [test_faces[i] for i in range(test_faces.shape[0])]
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="dict",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertTrue(equi_img.requires_grad)

    def test_e2c_dict_grad(self) -> None:
        dict_keys =  ["Front", "Right", "Back", "Left", "Up", "Down"]
        face_width = 512
        equi_img = torch.ones([3, face_width * 2, face_width * 4], requires_grad=True)
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="dict"
        )
        for i in dict_keys:
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])
        for i in dict_keys:
            self.assertTrue(cubic_img[i].requires_grad)

    def test_c2e_then_e2c_dict_grad(self) -> None:
        dict_keys =  ["Front", "Right", "Back", "Left", "Up", "Down"]
        face_width = 512
        test_faces_tensors = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        test_faces = {k: test_faces_tensors[i] for i, k in zip(range(6), dict_keys)}
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="dict"
        )
        for i in dict_keys:
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])
        for i in dict_keys:
            self.assertTrue(cubic_img[i].requires_grad)
