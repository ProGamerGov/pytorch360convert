from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F


def rotation_matrix(rad: torch.Tensor, ax: torch.Tensor) -> torch.Tensor:
    """
    Create a rotation matrix for a given angle and axis.

    Args:
        rad (torch.Tensor): Rotation angle in radians.
        ax (torch.Tensor): Rotation axis vector.

    Returns:
        torch.Tensor: 3x3 rotation matrix.
    """
    ax = ax / torch.sqrt((ax**2).sum())
    c = torch.cos(rad)
    s = torch.sin(rad)
    R = torch.diag(torch.tensor([c, c, c], dtype=ax.dtype, device=ax.device))
    R = R + (1.0 - c) * torch.ger(ax, ax)
    K = torch.tensor(
        [[0, -ax[2], ax[1]], [ax[2], 0, -ax[0]], [-ax[1], ax[0], 0]],
        dtype=ax.dtype,
        device=ax.device,
    )
    R = R + K * s
    return R


def xyzcube(
    face_w: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate cube coordinates for equirectangular projection.

    Args:
        face_w (int): Width of each cube face.
        device (torch.device, optional): Device to create tensor on. Defaults
            to torch.device('cpu').
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to
            torch.float32.


    Returns:
        torch.Tensor: Cube coordinates tensor of shape (face_w, face_w * 6, 3).
    """
    rng = torch.linspace(-0.5, 0.5, steps=face_w, dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(rng, -rng)  # shape (face_w, face_w)
    grid = torch.stack([grid_x, grid_y], dim=-1)  # (face_w, face_w, 2)

    out = torch.zeros((face_w, face_w * 6, 3), dtype=dtype, device=device)
    # Front
    out[:, 0 * face_w : 1 * face_w, 0:2] = grid
    out[:, 0 * face_w : 1 * face_w, 2] = 0.5
    # Right
    out[:, 1 * face_w : 2 * face_w, [2, 1]] = grid
    out[:, 1 * face_w : 2 * face_w, 0] = 0.5
    # Back
    out[:, 2 * face_w : 3 * face_w, 0:2] = grid
    out[:, 2 * face_w : 3 * face_w, 2] = -0.5
    # Left
    out[:, 3 * face_w : 4 * face_w, [2, 1]] = grid
    out[:, 3 * face_w : 4 * face_w, 0] = -0.5
    # Up
    out[:, 4 * face_w : 5 * face_w, [0, 2]] = grid
    out[:, 4 * face_w : 5 * face_w, 1] = 0.5
    # Down
    out[:, 5 * face_w : 6 * face_w, [0, 2]] = grid
    out[:, 5 * face_w : 6 * face_w, 1] = -0.5

    return out


def equirect_uvgrid(
    h: int,
    w: int,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate UV grid for equirectangular projection.

    Args:
        h (int): Height of the grid.
        w (int): Width of the grid.
        device (torch.device, optional): Device to create tensor on. Defaults
            to torch.device('cpu').
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to
            torch.float32.

    Returns:
        torch.Tensor: UV grid of shape (h, w, 2).
    """
    u = torch.linspace(-torch.pi, torch.pi, steps=w, dtype=dtype, device=device)
    v = torch.linspace(torch.pi, -torch.pi, steps=h, dtype=dtype, device=device) / 2
    grid_v, grid_u = torch.meshgrid(v, u)
    uv = torch.stack([grid_u, grid_v], dim=-1)
    return uv


def equirect_facetype(
    h: int, w: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """
    Determine face types for equirectangular projection.

    Args:
        h (int): Height of the grid.
        w (int): Width of the grid.
        device (torch.device, optional): Device to create tensor on. Defaults
            to torch.device('cpu').

    Returns:
        torch.Tensor: Face type tensor of shape (h, w) with integer face
            indices.
    """
    tp = (
        torch.arange(4, device=device)
        .repeat_interleave(w // 4)
        .unsqueeze(0)
        .repeat(h, 1)
    )
    tp = torch.roll(tp, shifts=3 * (w // 8), dims=1)

    # Prepare ceil mask
    mask = torch.zeros((h, w // 4), dtype=torch.bool, device=device)
    idx = torch.linspace(-torch.pi, torch.pi, w // 4, device=device) / 4
    idx = h // 2 - torch.round(torch.atan(torch.cos(idx)) * h / torch.pi).to(torch.long)
    for i, j in enumerate(idx):
        mask[:j, i] = True
    mask = torch.roll(torch.cat([mask] * 4, dim=1), shifts=3 * (w // 8), dims=1)

    tp[mask] = 4
    tp[torch.flip(mask, [0])] = 5
    return tp.to(torch.int32)


def xyzpers(
    h_fov: float,
    v_fov: float,
    u: float,
    v: float,
    out_hw: Tuple[int, int],
    in_rot: float,
    device: torch.device = torch.device("cpu"),
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """
    Generate perspective projection coordinates.

    Args:
        h_fov (torch.Tensor): Horizontal field of view in radians.
        v_fov (torch.Tensor): Vertical field of view in radians.
        u (float): Horizontal rotation angle in radians.
        v (float): Vertical rotation angle in radians.
        out_hw (Tuple[int, int]): Output height and width.
        in_rot (torch.Tensor): Input rotation angle in radians.
        device (torch.device, optional): Device to create tensor on. Defaults
            to torch.device('cpu').
        dtype (torch.dtype, optional): Data type of the tensor. Defaults to
            torch.float32.

    Returns:
        torch.Tensor: Perspective projection coordinates tensor.
    """
    h_fov = torch.tensor([h_fov], dtype=dtype, device=device)
    v_fov = torch.tensor([v_fov], dtype=dtype, device=device)
    u = torch.tensor([u], dtype=dtype, device=device)
    v = torch.tensor([v], dtype=dtype, device=device)
    in_rot = torch.tensor([in_rot], dtype=dtype, device=device)

    out = torch.ones((*out_hw, 3), dtype=dtype, device=device)
    x_max = torch.tan(h_fov / 2)
    y_max = torch.tan(v_fov / 2)
    y_range = torch.linspace(-y_max, y_max, steps=out_hw[0], dtype=dtype, device=device)
    x_range = torch.linspace(-x_max, x_max, steps=out_hw[1], dtype=dtype, device=device)
    grid_y, grid_x = torch.meshgrid(-y_range, x_range)
    out[..., 0] = grid_x
    out[..., 1] = grid_y

    Rx = rotation_matrix(v, torch.tensor([1.0, 0.0, 0.0], dtype=dtype, device=device))
    Ry = rotation_matrix(u, torch.tensor([0.0, 1.0, 0.0], dtype=dtype, device=device))
    Ri = rotation_matrix(
        in_rot, torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device) @ Rx @ Ry
    )

    # Apply R = Rx*Ry*Ri to each vector
    # like this: out * Rx * Ry * Ri (assuming row vectors)
    out = out @ Rx @ Ry @ Ri
    return out


def xyz2uv(xyz: torch.Tensor) -> torch.Tensor:
    """
    Transform cartesian (x, y, z) to spherical(r, u, v), and
    only outputs (u, v).

    Args:
        xyz (torch.Tensor): Input 3D coordinates tensor.

    Returns:
        torch.Tensor: UV coordinates tensor.
    """
    x = xyz[..., 0]
    y = xyz[..., 1]
    z = xyz[..., 2]
    u = torch.atan2(x, z)
    c = torch.sqrt(x**2 + z**2)
    v = torch.atan2(y, c)
    return torch.stack([u, v], dim=-1)


def uv2unitxyz(uv: torch.Tensor) -> torch.Tensor:
    """
    Convert UV coordinates to unit 3D Cartesian coordinates.

    Args:
        uv (torch.Tensor): Input UV coordinates tensor.

    Returns:
        torch.Tensor: Unit 3D coordinates tensor.
    """
    u = uv[..., 0]
    v = uv[..., 1]
    y = torch.sin(v)
    c = torch.cos(v)
    x = c * torch.sin(u)
    z = c * torch.cos(u)
    return torch.stack([x, y, z], dim=-1)


def uv2coor(uv: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Convert UV coordinates to image coordinates.

    Args:
        uv (torch.Tensor): Input UV coordinates tensor.
        h (int): Image height.
        w (int): Image width.

    Returns:
        torch.Tensor: Image coordinates tensor.
    """
    u = uv[..., 0]
    v = uv[..., 1]
    coor_x = (u / (2 * torch.pi) + 0.5) * w - 0.5
    coor_y = (-v / torch.pi + 0.5) * h - 0.5
    return torch.stack([coor_x, coor_y], dim=-1)


def coor2uv(coorxy: torch.Tensor, h: int, w: int) -> torch.Tensor:
    """
    Convert image coordinates to UV coordinates.

    Args:
        coorxy (torch.Tensor): Input image coordinates tensor.
        h (int): Image height.
        w (int): Image width.

    Returns:
        torch.Tensor: UV coordinates tensor.
    """
    coor_x = coorxy[..., 0]
    coor_y = coorxy[..., 1]
    u = ((coor_x + 0.5) / w - 0.5) * 2 * torch.pi
    v = -((coor_y + 0.5) / h - 0.5) * torch.pi
    return torch.stack([u, v], dim=-1)


def grid_sample_wrap(
    image: torch.Tensor,
    coor_x: torch.Tensor,
    coor_y: torch.Tensor,
    mode: str = "bilinear",
) -> torch.Tensor:
    """
    Sample from an image with wrapped horizontal coordinates.

    Args:
        image (torch.Tensor): Input image tensor of shape [H, W, C].
        coor_x (torch.Tensor): X coordinates for sampling.
        coor_y (torch.Tensor): Y coordinates for sampling.
        mode (str, optional): Sampling interpolation mode, 'nearest' or
            'bilinear'. Defaults to 'bilinear'.

    Returns:
        torch.Tensor: Sampled image tensor.
    """
    H, W, C = image.shape

    # coor_x, coor_y: [H_out, W_out]
    # We must create a grid for F.grid_sample:
    # grid_sample expects: input [N, C, H, W], grid [N,H_out, W_out, 2]
    # Normalized coords: x: [-1, 1], y: [-1, 1]
    # Handle wrapping horizontally: coor_x modulo W
    coor_x_wrapped = torch.remainder(coor_x, W)  # wrap horizontally
    coor_y_clamped = coor_y.clamp(min=0, max=H - 1)

    # Normalize
    grid_x = (coor_x_wrapped / (W - 1)) * 2 - 1
    grid_y = (coor_y_clamped / (H - 1)) * 2 - 1
    grid = torch.stack([grid_x, grid_y], dim=-1)  # [H_out, W_out, 2]

    # Permute image to NCHW
    img_t = image.permute(2, 0, 1).unsqueeze(0)  # [1,C,H,W]
    grid = grid.unsqueeze(0)  # [1,H_out,W_out,2]

    # grid_sample: note that the code samples using (y,x) order if
    # align_corners=False, we must be careful:
    # grid is defined as grid[:,:,:,0] = x, grid[:,:,:,1] = y,
    # PyTorch grid_sample expects grid in form (N, H_out, W_out,2),
    # with grid[:,:,:,0] = x and grid[:,:,:,1] = y

    if img_t.dtype == torch.float16 and img_t.device == torch.device("cpu"):
        sampled = F.grid_sample(
            img_t.float(),
            grid.float(),
            mode=mode,
            padding_mode="border",
            align_corners=True,
        ).half()
    else:
        sampled = F.grid_sample(
            img_t, grid, mode=mode, padding_mode="border", align_corners=True
        )

    # [1,C,H_out,W_out]
    sampled = sampled.squeeze(0).permute(1, 2, 0)  # [H_out, W_out, C]
    return sampled


def sample_equirec(
    e_img: torch.Tensor, coor_xy: torch.Tensor, order: int
) -> torch.Tensor:
    """
    Sample from an equirectangular image.

    Args:
        e_img (torch.Tensor): Equirectangular image tensor of shape [H, W, C].
        coor_xy (torch.Tensor): Sampling coordinates of shape
            [H_out, W_out, 2].
        order (int): Sampling interpolation order (0 for nearest, 1 for
            bilinear).

    Returns:
        torch.Tensor: Sampled image tensor.
    """
    mode = "bilinear" if order == 1 else "nearest"
    coor_x = coor_xy[..., 0]
    coor_y = coor_xy[..., 1]
    return grid_sample_wrap(e_img, coor_x, coor_y, mode=mode)


def sample_cubefaces(
    cube_faces: torch.Tensor,
    tp: torch.Tensor,
    coor_y: torch.Tensor,
    coor_x: torch.Tensor,
    order: int,
) -> torch.Tensor:
    """
    Sample from cube faces.

    Args:
        cube_faces (torch.Tensor): Cube faces tensor of shape
            [6, face_w, face_w, C].
        tp (torch.Tensor): Face type tensor.
        coor_y (torch.Tensor): Y coordinates for sampling.
        coor_x (torch.Tensor): X coordinates for sampling.
        order (int): Sampling interpolation order (0 for nearest,
            1 for bilinear).

    Returns:
        torch.Tensor: Sampled cube faces tensor.
    """
    # cube_faces: [6,face_w, face_w, C]
    # We must sample according to tp (face index), coor_y, coor_x
    # First we must flatten all faces into a single big image (like cube_h)
    # The original, code tries to do complicated padding and wrapping.
    # We'll try a simpler approach: we have tp that selects face.
    # We can do per-face sampling. Instead of map_coordinates
    # (tp, y, x), we know each pixel belongs to a certain face.

    # For differentiability and simplicity, let's do a trick:
    # Create a big image [face_w,face_w*6, C] (cube_h) and sample from it using
    # coor_x, coor_y and tp.
    cube_faces_mod = cube_faces.clone()
    cube_faces_mod[1] = torch.flip(cube_faces_mod[1], dims=[1])
    cube_faces_mod[2] = torch.flip(cube_faces_mod[2], dims=[1])
    cube_faces_mod[4] = torch.flip(cube_faces_mod[4], dims=[0])

    face_w = cube_faces_mod.shape[1]
    cube_h = torch.cat(
        [cube_faces_mod[i] for i in range(6)], dim=1
    )  # [face_w, face_w*6, C]

    # We need to map (tp, coor_y, coor_x) -> coordinates in cube_h
    # cube_h faces: 0:F, 1:R, 2:B, 3:L, 4:U, 5:D in order
    # If tp==0: x in [0, face_w-1] + offset 0
    # If tp==1: x in [0, face_w-1] + offset face_w
    # etc.

    # coor_x, coor_y are in face coordinates [0, face_w-1]
    # offset for face
    # x_offset = tp * face_w

    # Construct a single image indexing:
    # To handle tp indexing, let's create global_x = coor_x + tp * face_w
    # But tp might have shape (H_out,W_out)
    global_x = coor_x + tp.float() * face_w
    global_y = coor_y

    mode = "bilinear" if order == 1 else "nearest"

    return grid_sample_wrap(cube_h, global_x, global_y, mode=mode)


def cube_h2list(cube_h: torch.Tensor) -> List[torch.Tensor]:
    """
    Convert a horizontal cube representation to a list of cube faces.

    Args:
        cube_h (torch.Tensor): Horizontal cube representation tensor of shape
            [w, w*6, C].

    Returns:
        List[torch.Tensor]: List of cube face tensors in the order of:
            ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']
    """
    w = cube_h.shape[0]
    return [cube_h[:, i * w : (i + 1) * w, :] for i in range(6)]


def cube_list2h(cube_list: List[torch.Tensor]) -> torch.Tensor:
    """
    Convert a list of cube faces to a horizontal cube representation.

    Args:
        cube_list (List[torch.Tensor]): List of cube face tensors, in order of
            ['Front', 'Right', 'Back', 'Left', 'Up', 'Down']

    Returns:
        torch.Tensor: Horizontal cube representation tensor.
    """
    return torch.cat(cube_list, dim=1)


def cube_h2dict(
    cube_h: torch.Tensor,
    face_keys: List[str] = ["Front", "Right", "Back", "Left", "Up", "Down"],
) -> Dict[str, torch.Tensor]:
    """
    Convert a horizontal cube representation to a dictionary of cube faces.

    Order: F R B L U D
    dice layout: 3*face_w x 4*face_w

    Args:
        cube_h (torch.Tensor): Horizontal cube representation tensor of shape
            [w, w*6, C].
        face_keys (List[str], optional): List of face keys in order. Defaults
            to ["Front", "Right", "Back", "Left", "Up", "Down"].

    Returns:
        Dict[str, torch.Tensor]: Dictionary of cube faces with keys
            ["Front", "Right", "Back", "Left", "Up", "Down"].
    """
    cube_list = cube_h2list(cube_h)
    return dict(zip(face_keys, cube_list))


def cube_dict2h(
    cube_dict: Dict[str, torch.Tensor],
    face_keys: List[str] = ["Front", "Right", "Back", "Left", "Up", "Down"],
) -> torch.Tensor:
    """
    Convert a dictionary of cube faces to a horizontal cube representation.

    Args:
        cube_dict (Dict[str, torch.Tensor]): Dictionary of cube faces.
        face_keys (List[str], optional): List of face keys in order. Defaults
            to ["Front", "Right", "Back", "Left", "Up", "Down"].

    Returns:
        torch.Tensor: Horizontal cube representation tensor.
    """
    return cube_list2h([cube_dict[k] for k in face_keys])


def cube_h2dice(cube_h: torch.Tensor) -> torch.Tensor:
    """
    Convert a horizontal cube representation to a dice layout representation.

    Args:
        cube_h (torch.Tensor): Horizontal cube representation tensor of shape
            [w, w*6, C].

    Returns:
        torch.Tensor: Dice layout cube representation tensor of shape [w*3, w*4, C].
    """
    w = cube_h.shape[0]
    cube_dice = torch.zeros(
        (w * 3, w * 4, cube_h.shape[2]), dtype=cube_h.dtype, device=cube_h.device
    )
    cube_list = cube_h2list(cube_h)
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_list[i]
        if i in [1, 2]:
            face = torch.flip(face, dims=[1])
        if i == 4:
            face = torch.flip(face, dims=[0])
        face = torch.flip(face, dims=[0, 1])
        cube_dice[sy * w : (sy + 1) * w, sx * w : (sx + 1) * w] = face
    return cube_dice


def cube_dice2h(cube_dice: torch.Tensor) -> torch.Tensor:
    """
    Convert a dice layout representation to a horizontal cube representation.

    Args:
        cube_dice (torch.Tensor): Dice layout cube representation tensor of shape
            [w*3, w*4, C].

    Returns:
        torch.Tensor: Horizontal cube representation tensor of shape [w, w*6, C].
    """
    w = cube_dice.shape[0] // 3
    cube_h = torch.zeros(
        (w, w * 6, cube_dice.shape[2]), dtype=cube_dice.dtype, device=cube_dice.device
    )
    sxy = [(1, 1), (2, 1), (3, 1), (0, 1), (1, 0), (1, 2)]
    for i, (sx, sy) in enumerate(sxy):
        face = cube_dice[sy * w : (sy + 1) * w, sx * w : (sx + 1) * w]
        if i in [1, 2]:
            face = torch.flip(face, dims=[1])
        if i == 4:
            face = torch.flip(face, dims=[0])
        cube_h[:, i * w : (i + 1) * w] = face
        face = torch.flip(face, dims=[0, 1])
    return cube_h


def c2e(
    cubemap: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]],
    h: Optional[int] = None,
    w: Optional[int] = None,
    mode: str = "bilinear",
    cube_format: str = "dice",
    device: torch.device = torch.device("cpu"),
    channels_first: bool = True,
) -> torch.Tensor:
    """
    Convert a cubemap to an equirectangular projection.

    Args:
        cubemap (Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]):
            The input cubemap. If `cube_format` is set to 'list' or 'stack' of
            tensors, the faces should be arranged in the following order:
            ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']. If
            `cubemap_format` is set to 'dict', the dictionary keys should be
            ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom'].
        h (int, optional): Height of the output equirectangular image. If set
            to None, <cube_face_width> * 2 will be used.
        w (int, optional): Width of the output equirectangular image. If set
            to None, <cube_face_width> * 4 will be used.
        mode (str, optional): Sampling interpolation mode, 'nearest' or
            'bilinear'. Defaults to 'bilinear'.
        cube_format (str, optional): Output cubemap format. Defaults to 'dice'.
            Options are:
            - 'stack': Stack of 6 faces (torch.Tensor).
            - 'list': List of 6 faces (List[torch.Tensor]).
            - 'dict': Dictionary with keys pointing to face tensors:
                (Dict[str, torch.Tensor]).
            - 'dice': A cubemap in a 'dice' layout (torch.Tensor).
            - 'horizon': A cubemap in a 'horizon' layout (torch.Tensor).
        device (torch.device, optional): Device to create tensor on. Defaults
            to torch.device('cpu').
        channels_first (bool, optional): The channel format of e_img. Defaults
            to 'True' for channels first.

    Returns:
        torch.Tensor: Equirectangular projection tensor.

    Raises:
        NotImplementedError: If an unknown cube_format is provided.
    """

    if cube_format == "stack":
        assert (
            isinstance(cubemap, torch.Tensor)
            and len(cubemap.shape) == 4
            and cubemap.shape[0] == 6
        )
        cubemap = [cubemap[i] for i in range(cubemap.shape[0])]
        cube_format = "list"

    # Ensure input is in HWC format for processing
    if channels_first:
        if cube_format == "list" and isinstance(cubemap, (list, tuple)):
            cubemap = [r.permute(1, 2, 0) for r in cubemap]
        elif cube_format == "dict" and isinstance(cubemap, dict):
            cubemap = {k: v.permute(1, 2, 0) for k, v in cubemap.items()}
        elif cube_format in ["horizon", "dice"] and isinstance(cubemap, torch.Tensor):
            cubemap = cubemap.permute(1, 2, 0)
        else:
            raise NotImplementedError("unknown cube_format and cubemap type")

    order = 1 if mode == "bilinear" else 0

    if cube_format == "horizon" and isinstance(cubemap, torch.Tensor):
        assert cubemap.dim() == 3
        cube_h = cubemap
    elif cube_format == "list" and isinstance(cubemap, (list, tuple)):
        assert all([r.dim() == 3 for r in cubemap])
        cube_h = cube_list2h(cubemap)
    elif cube_format == "dict" and isinstance(cubemap, dict):
        assert all(v.dim() == 3 for k, v in cubemap.items())
        cube_h = cube_dict2h(cubemap)
    elif cube_format == "dice" and isinstance(cubemap, torch.Tensor):
        assert len(cubemap.shape) == 3
        cube_h = cube_dice2h(cubemap)
    else:
        raise NotImplementedError("unknown cube_format and cubemap type")
    assert isinstance(cube_h, torch.Tensor)  # Mypy wants this

    face_w = cube_h.shape[0]
    assert cube_h.shape[1] == face_w * 6

    h = face_w * 2 if not h else h
    w = face_w * 4 if not w else w

    assert w % 8 == 0

    uv = equirect_uvgrid(h, w, device=device)
    u, v = uv[..., 0], uv[..., 1]

    cube_faces = torch.stack(
        torch.split(cube_h, face_w, dim=1), dim=0
    )  # [6, face_w, face_w,C]

    tp = equirect_facetype(h, w, device=device)

    coor_x = torch.zeros((h, w), device=device)
    coor_y = torch.zeros((h, w), device=device)

    # front, right, back, left
    for i in range(4):
        mask = tp == i
        coor_x[mask] = 0.5 * torch.tan(u[mask] - torch.pi * i / 2)
        coor_y[mask] = -0.5 * torch.tan(v[mask]) / torch.cos(u[mask] - torch.pi * i / 2)

    # Up
    mask = tp == 4
    c = 0.5 * torch.tan(torch.pi / 2 - v[mask])
    coor_x[mask] = c * torch.sin(u[mask])
    coor_y[mask] = c * torch.cos(u[mask])

    # Down
    mask = tp == 5
    c = 0.5 * torch.tan(torch.pi / 2 - torch.abs(v[mask]))
    coor_x[mask] = c * torch.sin(u[mask])
    coor_y[mask] = -c * torch.cos(u[mask])

    coor_x = (torch.clamp(coor_x, -0.5, 0.5) + 0.5) * face_w
    coor_y = (torch.clamp(coor_y, -0.5, 0.5) + 0.5) * face_w

    C = cube_faces.shape[-1]
    # sample each channel:
    equirec_channels = []
    for i in range(C):
        face_chan = cube_faces[..., i]  # [6, face_w, face_w]
        # add channel dimension and sample from cube_faces:
        sampled = sample_cubefaces(cube_faces, tp, coor_y, coor_x, order)[:, :, i]
        equirec_channels.append(sampled)
    equirec = torch.stack(equirec_channels, dim=-1)

    # Convert back to CHW if required
    equirec = equirec.permute(2, 0, 1) if channels_first else equirec
    return equirec


def e2c(
    e_img: torch.Tensor,
    face_w: int = 256,
    mode: str = "bilinear",
    cube_format: str = "dice",
    channels_first: bool = True,
) -> Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
    """
    Convert an equirectangular image to a cubemap.

    Args:
        e_img (torch.Tensor): Input equirectangular image tensor of shape
            [C, H, W] or [H, W, C].
        face_w (int, optional): Width of each cube face. Defaults to 256.
        mode (str, optional): Sampling interpolation mode, 'nearest' or
            'bilinear'. Defaults to 'bilinear'.
        cube_format (str, optional): Output cubemap format. Defaults to 'dice'.
            The order for 'list' and 'stack' is expected to be:
            ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
            Full list of options are:
            - 'stack': Stack of 6 faces (torch.Tensor).
            - 'list': List of 6 faces (List[torch.Tensor]).
            - 'dict': Dictionary with keys pointing to face tensors:
                (Dict[str, torch.Tensor]). Keys are expected to be:
                ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
            - 'dice': A cubemap in a 'dice' layout (torch.Tensor).
            - 'horizon': A cubemap in a 'horizon' layout (torch.Tensor).
        channels_first (bool, optional): The channel format of e_img. Defaults
            to 'True' for channels first.

    Returns:
        Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]:
            The 'stack' and 'list' formats have faces in the order of:
            ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom'].
            The cubemap in the specified format:
            - 'stack': Stack of 6 faces (torch.Tensor).
            - 'list': List of 6 faces (List[torch.Tensor]).
            - 'dict': Dictionary with keys pointing to face tensors
              (Dict[str, torch.Tensor]). With keys:
              ['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']
            - 'dice': A cubemap in a 'dice' layout (torch.Tensor).
            - 'horizon': A cubemap in a 'horizon' layout (torch.Tensor).

    Raises:
        NotImplementedError: If an unknown cube_format is provided.
    """
    assert len(e_img.shape) == 3
    e_img = e_img.permute(1, 2, 0) if channels_first else e_img
    h, w = e_img.shape[:2]
    order = 1 if mode == "bilinear" else 0

    # returns [face_w, face_w*6, 3] in order
    # [Front, Right, Back, Left, Up, Down]
    xyz = xyzcube(face_w, device=e_img.device, dtype=e_img.dtype)
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)
    # Sample all channels:
    out_c = sample_equirec(e_img, coor_xy, order)  # [face_w, 6*face_w, C]
    # out_c shape: we did it directly for each pixel in the cube map

    result: Union[torch.Tensor, List[torch.Tensor], Dict[str, torch.Tensor]]
    if cube_format == "horizon":
        result = out_c
    elif cube_format == "list" or cube_format == "stack":
        result = cube_h2list(out_c)
    elif cube_format == "dict":
        result = cube_h2dict(out_c)
    elif cube_format == "dice":
        result = cube_h2dice(out_c)
    else:
        raise NotImplementedError("unknown cube_format")

    # Convert to CHW if required
    if channels_first:
        if cube_format == "list" or cube_format == "stack":
            assert isinstance(result, (list, tuple))
            result = [r.permute(2, 0, 1) for r in result]
        elif cube_format == "dict":
            assert isinstance(result, dict)
            result = {k: v.permute(2, 0, 1) for k, v in result.items()}
        elif cube_format in ["horizon", "dice"]:
            assert isinstance(result, torch.Tensor)
            result = result.permute(2, 0, 1)
    if cube_format == "stack" and isinstance(result, (list, tuple)):
        result = torch.stack(result)
    return result


def e2p(
    e_img: torch.Tensor,
    fov_deg: Union[float, Tuple[float, float]],
    u_deg: float,
    v_deg: float,
    out_hw: Tuple[int, int],
    in_rot_deg: float = 0,
    mode: str = "bilinear",
    channels_first: bool = True,
) -> torch.Tensor:
    """
    Convert an equirectangular image to a perspective projection.

    Args:
        e_img (torch.Tensor): Input equirectangular image tensor of shape
            [C, H, W] or [H, W, C].
        fov_deg (Union[float, Tuple[float, float]]): Field of view in degrees.
            Can be a single float or (h_fov, v_fov) tuple.
        u_deg (float): Horizontal rotation angle in degrees.
        v_deg (float): Vertical rotation angle in degrees.
        out_hw (Tuple[int, int]): Output image height and width.
        in_rot_deg (float, optional): Input rotation angle in degrees. Defaults
            to 0.
        mode (str, optional): Sampling interpolation mode. Defaults to 'bilinear'.
        channels_first (bool, optional): The channel format of e_img. Defaults
            to 'True' for channels first.

    Returns:
        torch.Tensor: Perspective projection image tensor.
    """
    assert len(e_img.shape) == 3
    # Ensure input is in HWC format for processing
    e_img = e_img.permute(1, 2, 0) if channels_first else e_img
    h, w = e_img.shape[:2]

    if isinstance(fov_deg, (list, tuple)):
        h_fov_rad = fov_deg[0] * torch.pi / 180
        v_fov_rad = fov_deg[1] * torch.pi / 180
    else:
        fov = fov_deg * torch.pi / 180
        h_fov_rad = fov
        v_fov_rad = fov

    in_rot = in_rot_deg * torch.pi / 180

    order = 1 if mode == "bilinear" else 0

    u = -u_deg * torch.pi / 180
    v = v_deg * torch.pi / 180
    xyz = xyzpers(
        h_fov_rad,
        v_fov_rad,
        u,
        v,
        out_hw,
        in_rot,
        device=e_img.device,
        dtype=e_img.dtype,
    )
    uv = xyz2uv(xyz)
    coor_xy = uv2coor(uv, h, w)

    pers_img = sample_equirec(e_img, coor_xy, order)

    # Convert back to CHW if required
    pers_img = pers_img.permute(2, 0, 1) if channels_first else pers_img
    return pers_img
