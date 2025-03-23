#!/usr/bin/env python3
import math
import random
import unittest

import numpy as np

import torch
from pytorch360convert.pytorch360convert import (
    _face_slice,
    _nchw2nhwc,
    _nhwc2nchw,
    _slice_chunk,
    c2e,
    coor2uv,
    cube_dice2h,
    cube_dict2h,
    cube_h2dice,
    cube_h2dict,
    cube_h2list,
    cube_list2h,
    e2c,
    e2e,
    e2p,
    equirect_facetype,
    equirect_uvgrid,
    grid_sample_wrap,
    pad_180_to_360,
    pad_cube_faces,
    rotation_matrix,
    sample_cubefaces,
    uv2coor,
    uv2unitxyz,
    xyz2uv,
    xyzcube,
    xyzpers,
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


def _create_dice_layout(
    faces: torch.Tensor, face_h: int = 512, face_w: int = 512
) -> torch.Tensor:
    H, W = face_h, face_w
    cube = torch.zeros((3, 3 * H, 4 * W))
    cube[:, 0 : 1 * H, W : 2 * W] = faces[4]
    cube[:, 1 * H : 2 * H, 0:W] = faces[3]
    cube[:, 1 * H : 2 * H, W : 2 * W] = faces[0]
    cube[:, 1 * H : 2 * H, 2 * W : 3 * W] = faces[1]
    cube[:, 1 * H : 2 * H, 3 * W : 4 * W] = faces[2]
    cube[:, 2 * H : 3 * H, W : 2 * W] = faces[5]
    return cube


def _get_c2e_4x4_exact_tensor() -> torch.Tensor:
    a = 0.4000000059604645
    b = 0.6000000238418579
    c = 0.507179856300354
    d = 0.0000
    e = 0.09061708301305771
    f = 0.20000000298023224
    g = 0.36016160249710083

    # Create the expected middle part for each of the 3 matrices
    middle = [a, a, b, b, b, c, d, d, d, e, f, f, f, g, a, a]
    middle = torch.tensor(middle).unsqueeze(0)  # Shape (1, 16)

    # Create the base output for the tensor
    expected_output = torch.zeros(8, 16)

    # Fill the first and last rows with the constants
    expected_output[0:2] = 0.800000011920929
    expected_output[6:8] = 1.0000

    # Fill the middle rows with the specific values
    expected_output[2:6] = middle

    # Exact values for the last row (row 7)
    last_row = [
        0.7569681406021118,
        0.8475320339202881,
        1.0,
        0.8708105087280273,
        0.8414928317070007,
        0.791767954826355,
        0.9714627265930176,
        0.6305789947509766,
        0.6305789947509766,
        0.5338935256004333,
        0.8733490109443665,
        0.6829856634140015,
        0.7416210174560547,
        0.19919204711914062,
        0.8475320935249329,
        0.7569681406021118,
    ]
    expected_output[5] = torch.tensor(last_row)

    # Now, create the 3 matrices by stacking the result
    expected_output = torch.stack([expected_output] * 3, dim=0)

    return expected_output


def _get_e2c_4x4_exact_tensor() -> torch.Tensor:
    f = [[1.0, 1.2951672077178955, 1.7048327922821045, 2.0]] * 4
    r = [[2.0, 2.2951672077178955, 2.7048325538635254, 3.0]] * 4
    b = [[3.0, 3.0, 3.0, 0.0]] * 4
    l = [[0.0, 0.29516735672950745, 0.7048328518867493, 1.0]] * 4
    u = [
        [0.0, 3.0, 3.0, 3.0],
        [0.29516735672950745, 0.0, 3.0, 2.7048325538635254],
        [0.7048328518867493, 1.0, 2.0, 2.2951672077178955],
        [1.0, 1.2951672077178955, 1.7048327922821045, 2.0],
    ]
    d = u[::-1]
    expected_out = torch.stack(
        [
            torch.tensor(f).repeat(3, 1, 1),
            torch.tensor(r).repeat(3, 1, 1),
            torch.tensor(b).repeat(3, 1, 1),
            torch.tensor(l).repeat(3, 1, 1),
            torch.tensor(u).repeat(3, 1, 1),
            torch.tensor(d).repeat(3, 1, 1),
        ]
    )
    return expected_out


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
        angle = torch.tensor([0.0])
        result = rotation_matrix(angle, axis)
        expected = torch.eye(3)
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

    def test_rotation_matrix_90deg(self) -> None:
        # Test 90-degree rotation around x-axis
        axis = torch.tensor([1.0, 0.0, 0.0])
        angle = torch.tensor([math.pi / 2])
        result = rotation_matrix(angle, axis)
        expected = torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
        torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-6)

        # Test rotation matrix properties
        # Should be orthogonal (R * R.T = I)
        result_t = result.t()
        identity = torch.mm(result, result_t)
        torch.testing.assert_close(identity, torch.eye(3), rtol=1e-6, atol=1e-6)

    def test_nhwc_to_nchw_channels_first(self) -> None:
        input_tensor = torch.rand(2, 3, 4, 5)
        converted_tensor = _nhwc2nchw(input_tensor)
        self.assertEqual(converted_tensor.shape, (2, 5, 3, 4))

    def test_nchw_to_nhwc_channels_first(self) -> None:
        input_tensor = torch.rand(2, 5, 3, 4)
        converted_tensor = _nchw2nhwc(input_tensor)
        self.assertEqual(converted_tensor.shape, (2, 3, 4, 5))

    def test_slice_chunk_default(self) -> None:
        index = 2
        width = 3
        offset = 0
        expected = torch.tensor([6, 7, 8], dtype=torch.long)
        result = _slice_chunk(index, width, offset)
        torch.testing.assert_close(result, expected)

    def test_slice_chunk_with_offset(self) -> None:
        # Test with a non-zero offset
        index = 2
        width = 3
        offset = 1
        expected = torch.tensor([7, 8, 9], dtype=torch.long)
        result = _slice_chunk(index, width, offset)
        torch.testing.assert_close(result, expected)

    def test_slice_chunk_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        index = 2
        width = 3
        offset = 0
        expected = torch.tensor([6, 7, 8], dtype=torch.long).cuda()
        result = _slice_chunk(index, width, offset, device=expected.device)
        torch.testing.assert_close(result, expected)
        self.assertTrue(result.is_cuda)

    def test_face_slice(self) -> None:
        # Test _face_slice, which internally calls _slice_chunk
        index = 2
        face_w = 3
        expected = torch.tensor([6, 7, 8], dtype=torch.long)
        result = _face_slice(index, face_w)
        torch.testing.assert_close(result, expected)

    def test_face_slice_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        # Test _face_slice, which internally calls _slice_chunk
        index = 2
        face_w = 3
        expected = torch.tensor([6, 7, 8], dtype=torch.long).cuda()
        result = _face_slice(index, face_w, device=expected.device)
        torch.testing.assert_close(result, expected)
        self.assertTrue(result.is_cuda)

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

    def test_cube_h2list(self) -> None:
        # Create a mock tensor with a shape [w, w*6, C]
        w = 3  # width of the cube face
        C = 2  # number of channels (e.g., RGB)
        cube_h = torch.randn(w, w * 6, C)  # Random tensor with dimensions [3, 18, 2]

        # Call the function
        result = cube_h2list(cube_h)

        # Assert that the result is a list of 6 tensors (one for each face)
        self.assertEqual(len(result), 6)

        # Assert each tensor has the correct shape [w, w, C]
        for tensor in result:
            self.assertEqual(tensor.shape, (w, w, C))

        # Ensure the shapes are sliced correctly
        for i in range(6):
            self.assertTrue(torch.equal(result[i], cube_h[:, i * w : (i + 1) * w, :]))

    def test_cube_h2dict(self) -> None:
        # Create a mock tensor with a shape [w, w*6, C]
        w = 3  # width of the cube face
        C = 2  # number of channels (e.g., RGB)
        cube_h = torch.randn(w, w * 6, C)  # Random tensor with dimensions [3, 18, 2]
        face_keys = ["Front", "Right", "Back", "Left", "Up", "Down"]

        # Call the function
        result = cube_h2dict(cube_h, face_keys)

        # Assert that the result is a dictionary with 6 entries
        self.assertEqual(len(result), 6)

        # Assert that the dictionary keys are correct
        self.assertEqual(list(result.keys()), face_keys)

        # Assert each tensor has the correct shape [w, w, C]
        for face in face_keys:
            self.assertEqual(result[face].shape, (w, w, C))

        # Check that the values correspond to the expected slices of the input tensor
        for i, face in enumerate(face_keys):
            self.assertTrue(
                torch.equal(result[face], cube_h[:, i * w : (i + 1) * w, :])
            )

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

        # Check sum
        self.assertEqual(result.sum().item(), 384.0)

        # Check dtype
        self.assertEqual(result.dtype, torch.int64)

    def test_equirect_facetype_large(self) -> None:
        h, w = 512 * 2, 512 * 4
        result = equirect_facetype(h, w)

        # Check shape
        self.assertEqual(result.shape, (h, w))

        # Check face type range (0-5 for 6 faces)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 5))

        # Check sum
        self.assertEqual(result.sum().item(), 6510864.0)

        # Check dtype
        self.assertEqual(result.dtype, torch.int64)

    def test_equirect_facetype_float64(self) -> None:
        h, w = 8, 16
        result = equirect_facetype(h, w, dtype=torch.float64)

        # Check shape
        self.assertEqual(result.shape, (h, w))

        # Check face type range (0-5 for 6 faces)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 5))

        # Check sum
        self.assertEqual(result.sum().item(), 384.0)

        # Check dtype
        self.assertEqual(result.dtype, torch.int64)

    def test_equirect_facetype_float16(self) -> None:
        h, w = 8, 16
        result = equirect_facetype(h, w, dtype=torch.float16)

        # Check shape
        self.assertEqual(result.shape, (h, w))

        # Check face type range (0-5 for 6 faces)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 5))

        # Check sum
        self.assertEqual(result.sum().item(), 384.0)

        # Check dtype
        self.assertEqual(result.dtype, torch.int64)

    def test_equirect_facetype_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        h, w = 8, 16
        result = equirect_facetype(h, w, device=torch.device("cuda"))

        # Check shape
        self.assertEqual(result.shape, (h, w))

        # Check face type range (0-5 for 6 faces)
        self.assertTrue(torch.all(result >= 0))
        self.assertTrue(torch.all(result <= 5))

        # Check sum
        self.assertEqual(result.sum().item(), 384.0)

        # Check dtype
        self.assertEqual(result.dtype, torch.int64)

        # Check cuda
        self.assertTrue(result.is_cuda)

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

    def test_grid_sample_wrap_cpu_float16(self) -> None:
        # Create test image
        h, w = 4, 8
        channels = 3
        image = torch.arange(h * w * channels, dtype=torch.float16)
        image = image.reshape(h, w, channels)

        # Test basic sampling
        coor_x = torch.tensor([[1.5, 2.5], [3.5, 4.5]], dtype=torch.float16)
        coor_y = torch.tensor([[1.5, 1.5], [2.5, 2.5]], dtype=torch.float16)

        # Test both interpolation modes
        result_bilinear = grid_sample_wrap(image, coor_x, coor_y, mode="bilinear")
        self.assertEqual(result_bilinear.dtype, torch.float16)

    def test_sample_cubefaces_cpu_float16(self) -> None:
        # Face type tensor (which face to sample)
        tp = torch.tensor([[0, 1], [2, 3]], dtype=torch.float16)  # Random face types

        # Coordinates for sampling
        coor_y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float16
        )  # y-coordinates
        coor_x = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float16
        )  # x-coordinates

        mode = "bilinear"

        # Call sample_cubefaces
        output = sample_cubefaces(
            torch.ones([6, 8, 8, 3], dtype=torch.float16), tp, coor_y, coor_x, mode
        )
        self.assertEqual(output.dtype, tp.dtype)

    def test_sample_cubefaces(self) -> None:
        # Face type tensor (which face to sample)
        tp = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32)  # Random face types

        # Coordinates for sampling
        coor_y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )  # y-coordinates
        coor_x = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )  # x-coordinates

        mode = "bilinear"

        # Call sample_cubefaces
        output = sample_cubefaces(torch.ones(6, 8, 8, 3), tp, coor_y, coor_x, mode)
        self.assertEqual(output.sum().item(), 12.0)

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
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        # assertTensorAlmostEqual(self, cubic_img, test_faces)

    def test_c2e_then_e2c_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        face_width = 512
        test_faces = _create_test_faces(face_width, face_width).cuda()
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
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.is_cuda)  # type: ignore[union-attr]
        # assertTensorAlmostEqual(self, cubic_img, test_faces)

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
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.requires_grad)  # type: ignore[union-attr]

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
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.requires_grad)  # type: ignore[union-attr]

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
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])  # type: ignore[index]
        for i in range(6):
            self.assertTrue(cubic_img[i].requires_grad)  # type: ignore[index]

    def test_c2e_then_e2c_list_grad(self) -> None:
        face_width = 512
        test_faces_tensors = torch.ones(
            [6, 3, face_width, face_width], requires_grad=True
        )
        test_faces = [test_faces_tensors[i] for i in range(test_faces_tensors.shape[0])]
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
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])  # type: ignore[index]
        for i in range(6):
            self.assertTrue(cubic_img[i].requires_grad)  # type: ignore[index]

    def test_c2e_dict_grad(self) -> None:
        dict_keys = ["Front", "Right", "Back", "Left", "Up", "Down"]
        face_width = 512
        test_faces_tensors = torch.ones(
            [6, 3, face_width, face_width], requires_grad=True
        )
        test_faces = {k: test_faces_tensors[i] for i, k in zip(range(6), dict_keys)}
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="dict",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertTrue(equi_img.requires_grad)

    def test_c2e_then_e2c_dict_grad(self) -> None:
        dict_keys = ["Front", "Right", "Back", "Left", "Up", "Down"]
        face_width = 512
        test_faces_tensors = torch.ones(
            [6, 3, face_width, face_width], requires_grad=True
        )
        test_faces = {k: test_faces_tensors[i] for i, k in zip(range(6), dict_keys)}
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="dict",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="dict"
        )
        for i in dict_keys:
            self.assertEqual(list(cubic_img[i].shape), [3, face_width, face_width])  # type: ignore
        for i in dict_keys:
            self.assertTrue(cubic_img[i].requires_grad)  # type: ignore

    def test_c2e_stack_nohw_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], requires_grad=True)
        equi_img = c2e(
            test_faces,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertTrue(equi_img.requires_grad)

    def test_sample_cubefaces_py360convert(self) -> None:
        try:
            import py360convert as p360
        except:
            raise unittest.SkipTest(
                "py360convert not installed, skipping sample_cubefaces test"
            )
        # Face type tensor (which face to sample)
        tp = torch.tensor([[0, 1], [2, 3]], dtype=torch.float32)  # Random face types

        # Coordinates for sampling
        coor_y = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )  # y-coordinates
        coor_x = torch.tensor(
            [[0.0, 1.0], [0.0, 1.0]], dtype=torch.float32
        )  # x-coordinates

        mode = "bilinear"

        # Call sample_cubefaces
        output = sample_cubefaces(torch.ones(6, 8, 8, 3), tp, coor_y, coor_x, mode)
        output_np = p360.sample_cubefaces(
            torch.ones(6, 8, 8).numpy(),
            tp.numpy(),
            coor_y.numpy(),
            coor_x.numpy(),
            mode,
        )
        self.assertEqual(output.sum(), output_np.sum() * 3)

    def test_c2e_py360convert(self) -> None:
        try:
            import py360convert as p360
        except:
            raise unittest.SkipTest("py360convert not installed, skipping c2e test")

        face_width = 512
        test_faces = _create_test_faces(face_width, face_width)
        test_faces = [test_faces[i] for i in range(test_faces.shape[0])]
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )
        test_faces_np = [t.permute(1, 2, 0).numpy() for t in test_faces]

        equi_img_np = p360.c2e(
            test_faces_np,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )
        equi_img_np_tensor = torch.from_numpy(equi_img_np).permute(2, 0, 1).float()
        assertTensorAlmostEqual(self, equi_img, equi_img_np_tensor, 2722.8169)

    def test_c2e_then_e2c_py360convert(self) -> None:
        try:
            import py360convert as p360
        except:
            raise unittest.SkipTest(
                "py360convert not installed, skipping c2e and e2c test"
            )

        face_width = 512
        test_faces = _create_test_faces(face_width, face_width)
        test_faces = [test_faces[i] for i in range(test_faces.shape[0])]
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )
        test_faces_np = [t.permute(1, 2, 0).numpy() for t in test_faces]

        equi_img_np = p360.c2e(
            test_faces_np,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="list",
        )

        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="dice"
        )

        cubic_img_np = p360.e2c(
            equi_img_np, face_w=face_width, mode="bilinear", cube_format="dice"
        )
        cubic_img_np_tensor = torch.from_numpy(cubic_img_np).permute(2, 0, 1).float()

        assertTensorAlmostEqual(self, cubic_img, cubic_img_np_tensor, 5858.8921)  # type: ignore[arg-type]

    def test_e2c_horizon_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4], requires_grad=True)
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="horizon"
        )
        self.assertEqual(list(cubic_img.shape), [3, face_width, face_width * 6])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.requires_grad)  # type: ignore[union-attr]

    def test_e2c_dice_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4], requires_grad=True)
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="dice"
        )
        self.assertEqual(list(cubic_img.shape), [3, face_width * 3, face_width * 4])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.requires_grad)  # type: ignore[union-attr]

    def test_c2e_then_e2c_dice_grad(self) -> None:
        face_width = 512
        test_faces = torch.ones([3, face_width * 3, face_width * 4], requires_grad=True)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="dice",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        cubic_img = e2c(
            equi_img, face_w=face_width, mode="bilinear", cube_format="dice"
        )
        self.assertEqual(list(cubic_img.shape), [3, face_width * 3, face_width * 4])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.requires_grad)  # type: ignore[union-attr]

    def test_e2p(self) -> None:
        # Create a simple test equirectangular image
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w))
        # Add some recognizable pattern
        e_img[0, :, :] = torch.linspace(0, 1, w).repeat(
            h, 1
        )  # Red gradient horizontally
        e_img[1, :, :] = (
            torch.linspace(0, 1, h).unsqueeze(1).repeat(1, w)
        )  # Green gradient vertically

        # Test basic perspective projection
        fov_deg = 90.0
        u_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, u_deg, v_deg, out_hw)

        # Check output shape
        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])

        # Test with different FOV
        narrow_fov = e2p(e_img, 45.0, u_deg, v_deg, out_hw)
        wide_fov = e2p(e_img, 120.0, u_deg, v_deg, out_hw)

        # Narrow FOV should have less variation in values than wide FOV
        self.assertTrue(torch.std(narrow_fov) < torch.std(wide_fov))

        # Test with rotation
        rotated = e2p(e_img, fov_deg, 90.0, v_deg, out_hw)  # 90 degrees right

        # Test with different output sizes
        large_output = e2p(e_img, fov_deg, u_deg, v_deg, (64, 64))
        self.assertEqual(list(large_output.shape), [channels, 64, 64])

        # Test with rectangular output
        rect_output = e2p(e_img, fov_deg, u_deg, v_deg, (32, 64))
        self.assertEqual(list(rect_output.shape), [channels, 32, 64])

        # Test with different FOV for height and width
        fov_hw = (90.0, 60.0)
        diff_fov = e2p(e_img, fov_hw, u_deg, v_deg, out_hw)
        self.assertEqual(list(diff_fov.shape), [channels, out_hw[0], out_hw[1]])

    def test_e2c_stack_face_w_none(self) -> None:
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4], requires_grad=True)
        cubic_img = e2c(test_faces, face_w=None, mode="bilinear", cube_format="stack")
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]

    def test_e2c_stack_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4]).cuda()
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertTrue(cubic_img.is_cuda)  # type: ignore[union-attr]

    def test_c2e_stack_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width]).cuda()
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertTrue(equi_img.is_cuda)

    def test_e2c_stack_float16(self) -> None:
        face_width = 512
        test_faces = torch.ones(
            [3, face_width * 2, face_width * 4], dtype=torch.float16
        )
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertEqual(cubic_img.dtype, torch.float16)  # type: ignore[union-attr]

    def test_e2c_stack_float64(self) -> None:
        face_width = 512
        test_faces = torch.ones(
            [3, face_width * 2, face_width * 4], dtype=torch.float64
        )
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertEqual(cubic_img.dtype, torch.float64)  # type: ignore[union-attr]

    def test_c2e_stack_float16(self) -> None:
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], dtype=torch.float16)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertEqual(equi_img.dtype, torch.float16)

    def test_c2e_stack_float64(self) -> None:
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], dtype=torch.float64)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertEqual(equi_img.dtype, torch.float64)

    def test_e2p_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        # Create a simple test equirectangular image
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w)).cuda()

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertTrue(result.is_cuda)

    def test_e2p_grad(self) -> None:
        # Create a simple test equirectangular image
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w), requires_grad=True)

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertTrue(result.requires_grad)

    def test_e2p_float16(self) -> None:
        # Create a simple test equirectangular image
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w), dtype=torch.float16)

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertEqual(result.dtype, torch.float16)

    def test_e2p_float64(self) -> None:
        # Create a simple test equirectangular image
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w), dtype=torch.float64)

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertEqual(result.dtype, torch.float64)

    def test_c2e_stack_1channel(self) -> None:
        channels = 1
        face_width = 512
        test_faces = torch.ones(
            [6, channels, face_width, face_width], dtype=torch.float64
        )
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(
            list(equi_img.shape), [channels, face_width * 2, face_width * 4]
        )

    def test_c2e_stack_4channels(self) -> None:
        channels = 4
        face_width = 512
        test_faces = torch.ones(
            [6, channels, face_width, face_width], dtype=torch.float64
        )
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(
            list(equi_img.shape), [channels, face_width * 2, face_width * 4]
        )

    def test_e2c_stack_1channel(self) -> None:
        channels = 1
        face_width = 512
        test_faces = torch.ones([channels, face_width * 2, face_width * 4])
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, channels, face_width, face_width])  # type: ignore[union-attr]

    def test_e2c_stack_4channels(self) -> None:
        channels = 4
        face_width = 512
        test_faces = torch.ones([channels, face_width * 2, face_width * 4])
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, channels, face_width, face_width])  # type: ignore[union-attr]

    def test_e2p_1channel(self) -> None:
        h, w = 64, 128
        channels = 1
        e_img = torch.zeros((channels, h, w))

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)
        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])

    def test_e2p_4channels(self) -> None:
        h, w = 64, 128
        channels = 4
        e_img = torch.zeros((channels, h, w))

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)
        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])

    def test_c2e_stack_jit(self) -> None:
        channels = 3
        face_width = 512
        test_faces = torch.ones(
            [6, channels, face_width, face_width], dtype=torch.float64
        )

        c2e_jit = torch.jit.script(c2e)
        equi_img = c2e_jit(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(
            list(equi_img.shape), [channels, face_width * 2, face_width * 4]
        )

    def test_e2c_stack_jit(self) -> None:
        channels = 3
        face_width = 512
        test_faces = torch.ones([channels, face_width * 2, face_width * 4])
        e2c_jit = torch.jit.script(e2c)
        cubic_img = e2c_jit(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, channels, face_width, face_width])

    def test_e2p_jit(self) -> None:
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w))

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        e2p_jit = torch.jit.script(e2p)

        result = e2p_jit(e_img, fov_deg, h_deg, v_deg, out_hw)
        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])

    def test_e2p_exact(self) -> None:
        channels = 3
        out_hw = (2, 4)
        x_input0 = torch.arange(1, 256).repeat(1, 256, 2).float()
        x_input1 = torch.arange(1, 256).repeat(1, 256, 2).float() * 2.0
        x_input2 = torch.arange(1, 256).repeat(1, 256, 2).float() * 4.0
        x_input = torch.cat([x_input0, x_input1, x_input2], 0)

        result = e2p(
            x_input,
            fov_deg=(40, 60),
            h_deg=10,
            v_deg=50,
            out_hw=out_hw,
            mode="bilinear",
        )

        a = torch.tensor(
            [
                [
                    183.03811645507812,
                    225.49948120117188,
                    58.833866119384766,
                    101.29522705078125,
                ],
                [
                    243.3969268798828,
                    5.628509521484375,
                    23.704801559448242,
                    40.9364013671875,
                ],
            ]
        )
        b = torch.tensor(
            [
                [
                    366.07623291015625,
                    450.99896240234375,
                    117.66773223876953,
                    202.5904541015625,
                ],
                [
                    486.7938537597656,
                    11.25701904296875,
                    47.409603118896484,
                    81.872802734375,
                ],
            ]
        )
        c = torch.tensor(
            [
                [
                    732.1524658203125,
                    901.9979248046875,
                    235.33546447753906,
                    405.180908203125,
                ],
                [
                    973.5877075195312,
                    22.5140380859375,
                    94.81920623779297,
                    163.74560546875,
                ],
            ]
        )
        expected_output = torch.stack([a, b, c])

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertTrue(torch.allclose(result, expected_output))

    def test_e2p_exact_batch(self) -> None:
        batch = 2
        channels = 3
        out_hw = (2, 4)
        x_input0 = torch.arange(1, 256).repeat(1, 256, 2).float()
        x_input1 = torch.arange(1, 256).repeat(1, 256, 2).float() * 2.0
        x_input2 = torch.arange(1, 256).repeat(1, 256, 2).float() * 4.0
        x_input = torch.cat([x_input0, x_input1, x_input2], 0).repeat(batch, 1, 1, 1)

        result = e2p(
            x_input,
            fov_deg=(40, 60),
            h_deg=10,
            v_deg=50,
            out_hw=out_hw,
            mode="bilinear",
        )

        a = torch.tensor(
            [
                [
                    183.03811645507812,
                    225.49948120117188,
                    58.833866119384766,
                    101.29522705078125,
                ],
                [
                    243.3969268798828,
                    5.628509521484375,
                    23.704801559448242,
                    40.9364013671875,
                ],
            ]
        )
        b = torch.tensor(
            [
                [
                    366.07623291015625,
                    450.99896240234375,
                    117.66773223876953,
                    202.5904541015625,
                ],
                [
                    486.7938537597656,
                    11.25701904296875,
                    47.409603118896484,
                    81.872802734375,
                ],
            ]
        )
        c = torch.tensor(
            [
                [
                    732.1524658203125,
                    901.9979248046875,
                    235.33546447753906,
                    405.180908203125,
                ],
                [
                    973.5877075195312,
                    22.5140380859375,
                    94.81920623779297,
                    163.74560546875,
                ],
            ]
        )
        expected_output = torch.stack([a, b, c]).repeat(batch, 1, 1, 1)

        self.assertEqual(list(result.shape), [batch, channels, out_hw[0], out_hw[1]])
        self.assertTrue(torch.allclose(result, expected_output))

    def test_c2e_stack_exact(self) -> None:
        expected_output = _get_c2e_4x4_exact_tensor()
        tile_w = 4
        x_input = _create_test_faces(tile_w, tile_w)
        output_cubic_tensor = c2e(x_input, mode="bilinear", cube_format="stack")
        self.assertTrue(torch.allclose(output_cubic_tensor, expected_output))

    def test_c2e_list_exact(self) -> None:
        expected_output = _get_c2e_4x4_exact_tensor()
        tile_w = 4
        x_input = _create_test_faces(tile_w, tile_w)
        dict_keys = ["Front", "Right", "Back", "Left", "Up", "Down"]
        x_input_list = [x_input[i] for i in range(6)]
        output_cubic_tensor = c2e(x_input_list, mode="bilinear", cube_format="list")
        self.assertTrue(torch.allclose(output_cubic_tensor, expected_output))

    def test_c2e_dict_exact(self) -> None:
        expected_output = _get_c2e_4x4_exact_tensor()
        tile_w = 4
        x_input = _create_test_faces(tile_w, tile_w)
        dict_keys = ["Front", "Right", "Back", "Left", "Up", "Down"]
        x_input_dict = {k: x_input[i] for i, k in zip(range(6), dict_keys)}
        output_cubic_tensor = c2e(x_input_dict, mode="bilinear", cube_format="dict")
        self.assertTrue(torch.allclose(output_cubic_tensor, expected_output))

    def test_c2e_horizon_exact(self) -> None:
        expected_output = _get_c2e_4x4_exact_tensor()
        tile_w = 4
        x_input = _create_test_faces(tile_w, tile_w)
        x_input_horizon = torch.cat([x_input[i] for i in range(6)], 2)
        output_cubic_tensor = c2e(
            x_input_horizon, mode="bilinear", cube_format="horizon"
        )
        self.assertTrue(torch.allclose(output_cubic_tensor, expected_output))

    def test_c2e_dice_exact(self) -> None:
        expected_output = _get_c2e_4x4_exact_tensor()
        tile_w = 4
        x_input = _create_test_faces(tile_w, tile_w)
        x_input = _create_dice_layout(x_input, tile_w, tile_w)
        output_cubic_tensor = c2e(x_input, mode="bilinear", cube_format="dice")
        self.assertTrue(torch.allclose(output_cubic_tensor, expected_output))

    def test_e2c_stack_exact(self) -> None:
        x_input = torch.arange(0, 4).repeat(3, 2, 1).float()
        output_tensor = e2c(x_input, face_w=4, mode="bilinear", cube_format="stack")
        expected_output = _get_e2c_4x4_exact_tensor()
        self.assertTrue(torch.allclose(output_tensor, expected_output))  # type: ignore[arg-type]

    def test_e2e_float16(self) -> None:
        channels = 4
        face_width = 512
        test_equi = torch.ones(
            [channels, face_width * 2, face_width * 4], dtype=torch.float16
        )
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, test_equi.dtype)

    def test_e2e_float64(self) -> None:
        channels = 4
        face_width = 512
        test_equi = torch.ones(
            [channels, face_width * 2, face_width * 4], dtype=torch.float64
        )
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, test_equi.dtype)

    def test_e2e_1channel(self) -> None:
        channels = 1
        face_width = 512
        test_equi = torch.ones([channels, face_width * 2, face_width * 4])
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, test_equi.dtype)

    def test_e2e_4channels(self) -> None:
        channels = 4
        face_width = 512
        test_equi = torch.ones([channels, face_width * 2, face_width * 4])
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, test_equi.dtype)

    def test_e2e_grad(self) -> None:
        channels = 3
        face_width = 512
        test_equi = torch.ones(
            [channels, face_width * 2, face_width * 4], requires_grad=True
        )
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, test_equi.dtype)
        self.assertTrue(equi_img.requires_grad)

    def test_e2e_gpu(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        channels = 3
        face_width = 512
        test_equi = torch.ones([channels, face_width * 2, face_width * 4]).cuda()
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, test_equi.dtype)
        self.assertTrue(equi_img.is_cuda)

    def test_e2e_jit(self) -> None:
        channels = 3
        face_width = 512
        test_equi = torch.ones([channels, face_width * 2, face_width * 4])
        e2e_jit = torch.jit.script(e2e)
        equi_img = e2e_jit(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))

    def test_e2e_batch(self) -> None:
        batch = 2
        channels = 4
        face_width = 512
        test_equi = torch.ones([batch, channels, face_width * 2, face_width * 4])
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))

    def test_e2e_exact(self) -> None:
        channels = 1
        face_width = 2

        w = face_width * 2
        h = face_width * 4
        test_equi = torch.arange(h * w * channels, dtype=torch.float32)
        test_equi = test_equi.reshape(channels, h, w)

        result = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(result.shape), list(test_equi.shape))

        expected_output = torch.tensor(
            [
                [
                    [
                        8.846327781677246,
                        7.38915491104126,
                        10.024271011352539,
                        11.211331367492676,
                    ],
                    [
                        9.07547378540039,
                        3.879967451095581,
                        12.109222412109375,
                        15.09968376159668,
                    ],
                    [
                        10.49519157409668,
                        2.552384614944458,
                        14.621706008911133,
                        19.000062942504883,
                    ],
                    [
                        12.717361450195312,
                        4.980112552642822,
                        17.24430274963379,
                        22.87851905822754,
                    ],
                    [
                        15.3826265335083,
                        8.457935333251953,
                        19.71427345275879,
                        26.661039352416992,
                    ],
                    [
                        18.184263229370117,
                        12.159286499023438,
                        21.694225311279297,
                        29.683902740478516,
                    ],
                    [
                        20.887590408325195,
                        15.91322135925293,
                        22.679805755615234,
                        29.112091064453125,
                    ],
                    [
                        20.43504524230957,
                        19.638233184814453,
                        22.215164184570312,
                        23.207149505615234,
                    ],
                ]
            ]
        )
        self.assertTrue(torch.allclose(result, expected_output))

    def test_e2e_exact_batch(self) -> None:
        batch = 2
        channels = 1
        face_width = 2

        w = face_width * 2
        h = face_width * 4
        test_equi = torch.arange(batch * h * w * channels, dtype=torch.float32)
        test_equi = test_equi.reshape(batch, channels, h, w)

        result = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(result.shape), list(test_equi.shape))

        expected_output = torch.tensor(
            [
                [
                    [
                        [
                            8.846327781677246,
                            7.38915491104126,
                            10.024271011352539,
                            11.211331367492676,
                        ],
                        [
                            9.07547378540039,
                            3.879967451095581,
                            12.109222412109375,
                            15.09968376159668,
                        ],
                        [
                            10.49519157409668,
                            2.552384614944458,
                            14.621706008911133,
                            19.000062942504883,
                        ],
                        [
                            12.717361450195312,
                            4.980112552642822,
                            17.24430274963379,
                            22.87851905822754,
                        ],
                        [
                            15.3826265335083,
                            8.457935333251953,
                            19.71427345275879,
                            26.661039352416992,
                        ],
                        [
                            18.184263229370117,
                            12.159286499023438,
                            21.694225311279297,
                            29.683902740478516,
                        ],
                        [
                            20.887590408325195,
                            15.91322135925293,
                            22.679805755615234,
                            29.112091064453125,
                        ],
                        [
                            20.43504524230957,
                            19.638233184814453,
                            22.215164184570312,
                            23.207149505615234,
                        ],
                    ]
                ],
                [
                    [
                        [
                            40.84632873535156,
                            39.38915252685547,
                            42.02427291870117,
                            43.21133041381836,
                        ],
                        [
                            41.07547378540039,
                            35.879966735839844,
                            44.109222412109375,
                            47.09968566894531,
                        ],
                        [
                            42.49519348144531,
                            34.5523796081543,
                            46.6217041015625,
                            51.000064849853516,
                        ],
                        [
                            44.71736145019531,
                            36.9801139831543,
                            49.244300842285156,
                            54.87852096557617,
                        ],
                        [
                            47.382625579833984,
                            40.45793533325195,
                            51.71427536010742,
                            58.66103744506836,
                        ],
                        [
                            50.18426513671875,
                            44.15928649902344,
                            53.6942253112793,
                            61.683902740478516,
                        ],
                        [
                            52.88758850097656,
                            47.9132194519043,
                            54.679805755615234,
                            61.112091064453125,
                        ],
                        [
                            52.43504333496094,
                            51.63823318481445,
                            54.21516418457031,
                            55.207149505615234,
                        ],
                    ]
                ],
            ]
        )
        self.assertEqual(result.shape, expected_output.shape)
        self.assertTrue(torch.allclose(result, expected_output))

    def test_c2e_stack_bfloat16(self) -> None:
        dtype = torch.bfloat16
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], dtype=dtype)
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertEqual(equi_img.dtype, dtype)

    def test_e2c_stack_bfloat16(self) -> None:
        dtype = torch.bfloat16
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4], dtype=dtype)
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertEqual(cubic_img.dtype, dtype)  # type: ignore[union-attr]

    def test_e2p_bfloat16(self) -> None:
        # Create a simple test equirectangular image
        dtype = torch.bfloat16
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w), dtype=dtype)

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertEqual(result.dtype, dtype)

    def test_e2e_bfloat16(self) -> None:
        dtype = torch.bfloat16
        channels = 4
        face_width = 512
        test_equi = torch.ones([channels, face_width * 2, face_width * 4], dtype=dtype)
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, dtype)

    def test_c2e_stack_bfloat16_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        dtype = torch.bfloat16
        face_width = 512
        test_faces = torch.ones([6, 3, face_width, face_width], dtype=dtype).cuda()
        equi_img = c2e(
            test_faces,
            face_width * 2,
            face_width * 4,
            mode="bilinear",
            cube_format="stack",
        )
        self.assertEqual(list(equi_img.shape), [3, face_width * 2, face_width * 4])
        self.assertEqual(equi_img.dtype, dtype)
        self.assertTrue(equi_img.is_cuda)

    def test_e2c_stack_bfloat16_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        dtype = torch.bfloat16
        face_width = 512
        test_faces = torch.ones([3, face_width * 2, face_width * 4], dtype=dtype).cuda()
        cubic_img = e2c(
            test_faces, face_w=face_width, mode="bilinear", cube_format="stack"
        )
        self.assertEqual(list(cubic_img.shape), [6, 3, face_width, face_width])  # type: ignore[union-attr]
        self.assertEqual(cubic_img.dtype, dtype)  # type: ignore[union-attr]
        self.assertTrue(cubic_img.is_cuda)  # type: ignore[union-attr]

    def test_e2p_bfloat16_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        # Create a simple test equirectangular image
        dtype = torch.bfloat16
        h, w = 64, 128
        channels = 3
        e_img = torch.zeros((channels, h, w), dtype=dtype).cuda()

        fov_deg = 90.0
        h_deg = 0.0
        v_deg = 0.0
        out_hw = (32, 32)

        result = e2p(e_img, fov_deg, h_deg, v_deg, out_hw)

        self.assertEqual(list(result.shape), [channels, out_hw[0], out_hw[1]])
        self.assertEqual(result.dtype, dtype)
        self.assertTrue(result.is_cuda)

    def test_e2e_bfloat16_cuda(self) -> None:
        if not torch.cuda.is_available():
            raise unittest.SkipTest("Skipping CUDA test due to not supporting CUDA.")
        dtype = torch.bfloat16
        channels = 4
        face_width = 512
        test_equi = torch.ones(
            [channels, face_width * 2, face_width * 4], dtype=dtype
        ).cuda()
        equi_img = e2e(
            test_equi,
            h_deg=45,
            v_deg=45,
            roll=25,
            mode="bilinear",
        )
        self.assertEqual(list(equi_img.shape), list(test_equi.shape))
        self.assertEqual(equi_img.dtype, dtype)
        self.assertTrue(equi_img.is_cuda)

    def test_pad_180_to_360_channels_first(self) -> None:
        e_img = torch.ones(3, 4, 4)  # [C, H, W] = [3, 4, 4]
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
        self.assertEqual(padded_img.shape, (3, 4, 8))
        self.assertTrue(torch.all(padded_img[:, :, 0] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, -1] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, 2:6] == 1.0))

    def test_pad_180_to_360_channels_last(self) -> None:
        e_img = torch.ones(4, 4, 3)  # [H, W, C] = [4, 4, 3]
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=False)
        self.assertEqual(padded_img.shape, (4, 8, 3))
        self.assertTrue(torch.all(padded_img[:, 0, :] == 0.0))
        self.assertTrue(torch.all(padded_img[:, -1, :] == 0.0))
        self.assertTrue(torch.all(padded_img[:, 2:6, :] == 1.0))

    def test_pad_180_to_360_batch(self) -> None:
        e_img = torch.ones(2, 3, 4, 4)  # [B, C, H, W] = [2, 3, 4, 4]
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
        self.assertEqual(padded_img.shape, (2, 3, 4, 8))
        self.assertTrue(torch.all(padded_img[:, :, :, 0] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, :, -1] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, :, 2:6] == 1.0))

    def test_pad_180_to_360_gpu(self) -> None:
        if torch.cuda.is_available():
            e_img = torch.ones(3, 4, 4, device="cuda")  # [C, H, W] = [3, 4, 4]
            padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
            self.assertTrue(padded_img.is_cuda)
            self.assertEqual(padded_img.shape, (3, 4, 8))
            self.assertTrue(torch.all(padded_img[:, :, 0] == 0.0))
            self.assertTrue(torch.all(padded_img[:, :, -1] == 0.0))
            self.assertTrue(torch.all(padded_img[:, :, 2:6] == 1.0))

    def test_pad_180_to_360_float16(self) -> None:
        e_img = torch.ones(3, 4, 4, dtype=torch.float16)
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
        self.assertEqual(padded_img.shape, (3, 4, 8))
        self.assertEqual(padded_img.dtype, torch.float16)
        self.assertTrue(torch.all(padded_img[:, :, 0] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, -1] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, 2:6] == 1.0))

    def test_pad_180_to_360_float64(self) -> None:
        e_img = torch.ones(3, 4, 4, dtype=torch.float64)
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
        self.assertEqual(padded_img.shape, (3, 4, 8))
        self.assertEqual(padded_img.dtype, torch.float64)
        self.assertTrue(torch.all(padded_img[:, :, 0] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, -1] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, 2:6] == 1.0))

    def test_pad_180_to_360_bfloat16(self) -> None:
        e_img = torch.ones(3, 4, 4, dtype=torch.bfloat16)
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
        self.assertEqual(padded_img.shape, (3, 4, 8))
        self.assertEqual(padded_img.dtype, torch.bfloat16)
        self.assertTrue(torch.all(padded_img[:, :, 0] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, -1] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, 2:6] == 1.0))

    def test_pad_180_to_360_gradient(self) -> None:
        e_img = torch.ones(3, 4, 4, dtype=torch.float32, requires_grad=True)
        padded_img = pad_180_to_360(e_img, fill_value=0.0, channels_first=True)
        self.assertTrue(padded_img.requires_grad)

    def test_pad_180_to_360_jit(self) -> None:
        e_img = torch.ones(3, 4, 4)  # [C, H, W] = [3, 4, 4]
        pad_180_to_360_jit = torch.jit.script(pad_180_to_360)
        padded_img = pad_180_to_360_jit(e_img, fill_value=0.0, channels_first=True)
        self.assertEqual(padded_img.shape, (3, 4, 8))
        self.assertTrue(torch.all(padded_img[:, :, 0] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, -1] == 0.0))
        self.assertTrue(torch.all(padded_img[:, :, 2:6] == 1.0))

    def _prepare_test_data_cube_padding(
        self,
        height: int = 4,
        width: int = 4,
        channels: int = 3,
        dtype: torch.dtype = torch.float32,
        device: str = "cpu",
        requires_grad: bool = False,
    ) -> torch.Tensor:
        """Helper to create test cube faces with specific configuration"""
        # Create test faces where each face has a unique value
        cube_faces = torch.zeros(
            (6, height, width, channels), dtype=dtype, device=device
        )

        # Set each face to a unique value for easy identification
        for face_idx in range(6):
            cube_faces[face_idx, :, :, :] = face_idx / 5.0  # Normalize to [0, 1]

        if requires_grad:
            cube_faces.requires_grad_(True)

        return cube_faces

    def test_pad_cube_faces_shape(self) -> None:
        """Test that the output shape is correct"""
        # Use equal height and width to avoid dimension mismatch when flipping
        height, width, channels = 4, 4, 3
        cube_faces = self._prepare_test_data_cube_padding(height, width, channels)

        padded_faces = pad_cube_faces(cube_faces)

        self.assertEqual(padded_faces.shape, (6, height + 2, width + 2, channels))

    def test_pad_cube_faces_jit(self) -> None:
        """Test that the function works with torch.jit.script"""
        cube_faces = self._prepare_test_data_cube_padding(4, 4, 3)

        # Script the function
        pad_cube_faces_jit = torch.jit.script(pad_cube_faces)

        # Run the scripted function
        padded_faces = pad_cube_faces_jit(cube_faces)

        # Verify output shape
        self.assertEqual(padded_faces.shape, (6, 6, 6, 3))

        # Compare with non-scripted function
        expected_output = pad_cube_faces(cube_faces)
        self.assertTrue(torch.allclose(padded_faces, expected_output))

    def test_pad_cube_faces_gradient(self) -> None:
        """Test that gradients flow through the function"""
        cube_faces = self._prepare_test_data_cube_padding(requires_grad=True)

        padded_faces = pad_cube_faces(cube_faces)

        # Check that requires_grad is preserved
        self.assertTrue(padded_faces.requires_grad)

        # Test gradient flow by computing a loss and backpropagating
        loss = padded_faces.sum()
        loss.backward()

        # Verify that the gradient exists and is not zero
        self.assertIsNotNone(cube_faces.grad)
        self.assertFalse(
            torch.allclose(cube_faces.grad, torch.zeros_like(cube_faces.grad))  # type: ignore[arg-type]
        )

    def test_pad_cube_faces_cuda(self) -> None:
        """Test that the function works on CUDA device if available"""
        if not torch.cuda.is_available():
            self.skipTest("CUDA not available")

        cube_faces = self._prepare_test_data_cube_padding(device="cuda")

        padded_faces = pad_cube_faces(cube_faces)

        # Check that the output is on the correct device
        self.assertTrue(padded_faces.is_cuda)

        # Verify shape
        self.assertEqual(padded_faces.shape, (6, 6, 6, 3))

    def test_pad_cube_faces_float16(self) -> None:
        """Test that the function works with float16 precision"""
        cube_faces = self._prepare_test_data_cube_padding(dtype=torch.float16)

        padded_faces = pad_cube_faces(cube_faces)

        # Check that dtype is preserved
        self.assertEqual(padded_faces.dtype, torch.float16)

    def test_pad_cube_faces_float32(self) -> None:
        """Test that the function works with float32 precision"""
        cube_faces = self._prepare_test_data_cube_padding(dtype=torch.float32)

        padded_faces = pad_cube_faces(cube_faces)

        # Check that dtype is preserved
        self.assertEqual(padded_faces.dtype, torch.float32)

    def test_pad_cube_faces_float64(self) -> None:
        """Test that the function works with float64 precision"""
        cube_faces = self._prepare_test_data_cube_padding(dtype=torch.float64)

        padded_faces = pad_cube_faces(cube_faces)

        # Check that dtype is preserved
        self.assertEqual(padded_faces.dtype, torch.float64)

    def test_pad_cube_faces_bfloat16(self) -> None:
        """Test that the function works with bfloat16 precision"""
        # Skip if bfloat16 is not supported
        if not hasattr(torch, "bfloat16"):
            self.skipTest("bfloat16 not supported in this PyTorch version")

        cube_faces = self._prepare_test_data_cube_padding(dtype=torch.bfloat16)

        padded_faces = pad_cube_faces(cube_faces)

        # Check that dtype is preserved
        self.assertEqual(padded_faces.dtype, torch.bfloat16)

    def test_pad_cube_faces_exact_values(self) -> None:
        """Test the exact values of padded faces for a small input"""
        # Define small cube faces with easily identifiable values
        cube_faces = torch.zeros((6, 2, 2, 1))

        # FRONT=0, RIGHT=1, BACK=2, LEFT=3, UP=4, DOWN=5
        for face_idx in range(6):
            # Set each face to its index value for easier verification
            cube_faces[face_idx, :, :, :] = face_idx

        padded_faces = pad_cube_faces(cube_faces)

        # Verify central values (should be unchanged)
        for face_idx in range(6):
            self.assertTrue(
                torch.all(padded_faces[face_idx, 1:-1, 1:-1, :] == face_idx)
            )

        # Verify specific border values based on the padding logic
        # These assertions check that the padding values come from the correct neighboring faces

        # Check FRONT face padding
        self.assertTrue(torch.all(padded_faces[0, 0, 1:-1, :] == 4))  # Top from UP
        self.assertTrue(
            torch.all(padded_faces[0, -1, 1:-1, :] == 5)
        )  # Bottom from DOWN
        self.assertTrue(torch.all(padded_faces[0, 1:-1, 0, :] == 3))  # Left from LEFT
        self.assertTrue(
            torch.all(padded_faces[0, 1:-1, -1, :] == 1)
        )  # Right from RIGHT

        # Check additional values for other faces to verify correct padding
        # RIGHT face
        self.assertTrue(torch.all(padded_faces[1, 1:-1, 0, :] == 0))  # Left from FRONT

        # BACK face
        self.assertTrue(torch.all(padded_faces[2, 1:-1, -1, :] == 3))  # Right from LEFT

        # UP face
        self.assertTrue(
            torch.all(padded_faces[4, -1, 1:-1, :] == 0)
        )  # Bottom from FRONT
