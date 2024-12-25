# 📷 PyTorch 360° Image Conversion Toolkit

[![PyPI - Version](https://img.shields.io/pypi/v/pytorch360convert)](https://pypi.org/project/pytorch360convert/)


## Overview

This PyTorch-based library provides powerful and differentiable image transformation utilities for converting between different panoramic image formats:

- **Equirectangular (360°) Images** 
- **Cubemap Representations**
- **Perspective Projections**

Built as an improved PyTorch implementation of the original [py360convert](https://github.com/sunset1995/py360convert) project, this library offers flexible, CPU & GPU-accelerated functions.


<div align="left">
 <img src="examples/basic_equirectangular.png" width="710px">
</div>

* Equirectangular format


<div align="left">
 <img src="examples/basic_dice_cubemap.png" width="710px">
</div>

* Cubemap 'dice' format


## 🔧 Requirements

- Python 3.7+
- PyTorch


## 📦 Installation

You can easily install the library using pip:

```bash
pip install pytorch360convert
```

Or you can install it from source like this:

```bash
pip install torch
```

Then clone the repository:

```bash
git clone https://github.com/ProGamerGov/pytorch360convert.git
cd pytorch360convert
pip install .
```


## 🚀 Key Features

- Lossless conversion between image formats.
- Supports different cubemap input formats (horizon, list, dict, dice).
- Configurable sampling modes (bilinear, nearest).
- Supports different dtypes (float16, float32, float64).
- CPU support.
- GPU acceleration.
- Differentiable transformations for deep learning pipelines.
- [TorchScript](https://pytorch.org/docs/stable/jit.html) (JIT) support.


## 💡 Usage Examples


### Helper Functions

First we'll setup some helper functions:

```bash
pip install torchvision pillow
```


```python
import torch
from torchvision.transforms import ToTensor, ToPILImage
from PIL import Image

def load_image_to_tensor(image_path: str) -> torch.Tensor:
    """Load an image as a PyTorch tensor."""
    return ToTensor()(Image.open(image_path).convert('RGB'))

def save_tensor_as_image(tensor: torch.Tensor, save_path: str) -> None:
    """Save a PyTorch tensor as an image."""
    ToPILImage()(tensor).save(save_path)

```

### Equirectangular to Cubemap Conversion

Converting equirectangular images into cubemaps is easy. For simplicity, we'll use the 'dice' format, which places all cube faces into a single 4x3 grid image.

```python
from pytorch360convert import e2c

# Load equirectangular image (3, 1376, 2752)
equi_image = load_image_to_tensor("examples/example_world_map_equirectangular.jpg")
face_w = equi_image.shape[2] // 4  # 2752 / 4 = 688

# Convert to cubemap (dice format)
cubemap = e2c(
    equi_image,                   # CHW format
    face_w=face_w,                # Width of each cube face
    mode='bilinear',              # Sampling interpolation
    cube_format='dice'            # Output cubemap layout
)

# Save cubemap faces
save_tensor_as_image(cubemap, "dice_cubemap.jpg")
```

| Equirectangular Input | Cubemap 'Dice' Output |
| :---: | :----: |
| ![](examples/example_world_map_equirectangular.jpg) | ![](examples/example_world_map_dice_cubemap.jpg) |

| Cubemap 'Horizon' Output |
| :---: |
| ![](examples/example_world_map_horizon_cubemap.jpg) |

### Cubemap to Equirectangular Conversion

We can also convert cubemaps into equirectangular images, like so.

```python
from pytorch360convert import c2e

# Load cubemap in 'dice' format
cubemap = load_image_to_tensor("dice_cubemap.jpg")

# Convert cubemap back to equirectangular
equirectangular = c2e(
    cubemap,              # Cubemap tensor(s)
    mode='bilinear',      # Sampling interpolation
    cube_format='dice'    # Input cubemap layout
)

save_tensor_as_image(equirectangular, "equirectangular.jpg")
```

### Equirectangular to Perspective Projection

```python
from pytorch360convert import e2p

# Load equirectangular input
equi_image = load_image_to_tensor("examples/example_world_map_equirectangular.jpg")

# Extract perspective view from equirectangular image
perspective_view = e2p(
    equi_image,                   # Equirectangular image
    fov_deg=(70, 60),             # Horizontal and vertical FOV
    h_deg=260,                    # Horizontal rotation
    w_deg=50,                     # Vertical rotation
    out_hw=(512, 768),            # Output image dimensions
    mode='bilinear'               # Sampling interpolation
)

save_tensor_as_image(perspective_view, "perspective.jpg")
```

| Equirectangular Input | Perspective Output |
| :---: | :----: |
| ![](examples/example_world_map_equirectangular.jpg) | ![](examples/example_world_map_perspective.jpg) |


## 📚 Basic Functions

### `e2c(e_img, face_w=256, mode='bilinear', cube_format='dice')`
Converts an equirectangular image to a cubemap projection.

- **Parameters**:
  - `e_img` (torch.Tensor): Equirectangular CHW image tensor.
  - `face_w` (int, optional): Cube face width. Default: `256`.
  - `mode` (str, optional): Sampling interpolation mode. Options are `bilinear`, `bicubic`, and `nearest`. Default: `bilinear`
  - `cube_format` (str, optional): The desired output cubemap format. Options are `dict`, `list`, `horizon`, `stack`, and `dice`. Default: `dice`
    - `stack` (torch.Tensor): Stack of 6 faces, in the order of: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
    - `list` (list of torch.Tensor): List of 6 faces, in the order of: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
    - `dict` (dict of torch.Tensor): Dictionary with keys pointing to face tensors. Keys are: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
    - `dice` (torch.Tensor): A cubemap in a 'dice' layout.
    - `horizon` (torch.Tensor): A cubemap in a 'horizon' layout, a 1x6 grid in the order: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
  - `channels_first` (bool, optional): Input cubemap channel format (CHW or HWC). Defaults to the PyTorch CHW standard of `True`.

- **Returns**: Cubemap representation of the input image as a tensor, list of tensors, or dict or tensors.

### `c2e(cubemap, h, w, mode='bilinear', cube_format='dice')`
Converts a cubemap projection to an equirectangular image.

- **Parameters**:
  - `cubemap` (torch.Tensor, list of torch.Tensor, or dict of torch.Tensor): Cubemap image tensor, list of tensors, or dict of tensors. Note that tensors should be in the shape of: `CHW`, except for when `cube_format = 'stack'`, in which case a batch dimension is present. Inputs should match the corresponding `cube_format`.
  - `h` (int, optional): Output image height. If set to None, `<cube_face_width> * 2` will be used. Default: `None`.
  - `w` (int, optional): Output image width. If set to None, `<cube_face_width> * 4` will be used. Default: `None`.
  - `mode` (str, optional): Sampling interpolation mode. Options are `bilinear`, `bicubic`, and `nearest`. Default: `bilinear`
  - `cube_format` (str, optional): Input cubemap format. Options are `dict`, `list`, `horizon`, `stack`, and `dice`. Default: `dice`
    - `stack` (torch.Tensor): Stack of 6 faces, in the order of: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
    - `list` (list of torch.Tensor): List of 6 faces, in the order of: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
    - `dict` (dict of torch.Tensor): Dictionary with keys pointing to face tensors. Keys are expected to be: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
    - `dice` (torch.Tensor): A cubemap in a 'dice' layout.
    - `horizon` (torch.Tensor): A cubemap in a 'horizon' layout, a 1x6 grid in the order of: ['Front', 'Right', 'Back', 'Left', 'Up', 'Down'].
  - `channels_first` (bool, optional): Input cubemap channel format (CHW or HWC). Defaults to the PyTorch CHW standard of `True`.
     
- **Returns**: Equirectangular projection of the input cubemap as a tensor.

### `e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear')`
Extracts a perspective view from an equirectangular image.

- **Parameters**:
  - `e_img` (torch.Tensor): Equirectangular CHW image tensor.
  - `fov_deg` (float or tuple): Field of view in degrees. If a single value is provided, it will be used for both horizontal and vertical degrees. If using a tuple, values are expected to be in following format: (h_fov_deg, v_fov_deg).
  - `h_deg` (float): Horizontal viewing angle in range [-pi, pi]. (- Left / + Right).
  - `w_deg` (float): Vertical viewing angle in range [-pi/2, pi/2]. (- Down/ + Up).
  - `out_hw` (tuple): Output image dimensions in the shape of '(height, width)'.
  - `in_rot_deg` (float, optional): Inplane rotation angle. Default: `0`
  - `mode` (str, optional): Sampling interpolation mode. Options are `bilinear`, `bicubic`, and `nearest`. Default: `bilinear`
  - `channels_first` (bool, optional): Input cubemap channel format (CHW or HWC). Defaults to the PyTorch CHW standard of `True`.

- **Returns**: Perspective view of the equirectangular image as a tensor.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🔬 Citation

If you use this library in your research or project, please refer to the included [CITATION.cff](CITATION.cff) file or cite it as follows:

### BibTeX
```bibtex
@misc{egan2024pytorch360convert,
  title={PyTorch 360° Image Conversion Toolkit},
  author={Egan, Ben},
  year={2024},
  publisher={GitHub},
  howpublished={\url{https://github.com/ProGamerGov/pytorch360convert}}
}
```

### APA Style
```
Egan, B. (2024). PyTorch 360° Image Conversion Toolkit [Computer software]. GitHub. https://github.com/ProGamerGov/pytorch360convert
```
