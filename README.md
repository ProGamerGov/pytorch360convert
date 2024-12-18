# 📷 PyTorch 360° Image Conversion Toolkit

[![PyPI - Version](https://img.shields.io/pypi/v/pytorch360convert)](https://pypi.org/project/pytorch360convert/)


## Overview

This PyTorch-based library provides powerful and differentiable image transformation utilities for converting between different panoramic image formats:

- **Equirectangular (360°) Images** 
- **Cubemap Representations**
- **Perspective Projections**

Built as an improved PyTorch implementation of the original [py360convert](https://github.com/sunset1995/py360convert) project, this library offers flexible, CPU & GPU-accelerated functions.


<div align="left">
 <img src="https://github.com/ProGamerGov/pytorch360convert/blob/main/examples/output_equirectangular.jpg?raw=true" width="710px">
</div>

* Equirectangular format


<div align="left">
 <img src="https://github.com/ProGamerGov/pytorch360convert/blob/main/examples/output_cubic.jpg?raw=true" width="710px">
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

Clone the repository:

```bash
git clone https://github.com/ProGamerGov/pytorch360convert.git
cd pytorch360convert
pip install .
```


## 🚀 Key Features

- Lossless conversion between image formats.
- Supports different cubemap input formats (horizon, list, dict, dice).
- Configurable sampling modes (bilinear, nearest).
- CPU and torch.float16 support.
- GPU acceleration.
- Differentiable transformations for deep learning pipelines.


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

Coverting equirectangular images into cubemaps is easy. For simplicitly, we'll use the 'dice' format, which places all cube faces into a single 4x3 grid image.

```python
from pytorch360convert import e2c

# Load equirectangular image
equi_image = load_image_to_tensor("360_panorama.jpg")

# Convert to cubemap (dice format)
cubemap = e2c(
    equi_image,                   # CHW format
    face_w=1024,                  # Width of each cube face
    mode='bilinear',              # Sampling interpolation
    cube_format='dice'            # Output cubemap layout
)

# Save cubemap faces
save_tensor_as_image(cubemap, "cubemap.jpg")
```

### Cubemap to Equirectangular Conversion

We can also convert cubemaps into equirectangular images, like so. Note that we use the same cubemap we created above and the same cubemap format used to make it.

```python
from pytorch360convert import c2e

# Load cubemap in 'dice' format
equi_image = load_image_to_tensor("cubemap.jpg")

# Convert cubemap back to equirectangular
equirectangular = c2e(
    cubemap,              # Cubemap tensor(s)
    h=2048,               # Output height
    w=4096,               # Output width
    mode='bilinear',      # Sampling interpolation
    cube_format='dice'    # Input cubemap layout
)

save_tensor_as_image(equirectangular, "equirectangular.jpg")
```

### Perspective Projection from Equirectangular

```python
from pytorch360convert import e2p

# Extract perspective view from equirectangular image
perspective_view = e2p(
    equi_image,                   # Equirectangular image
    fov_deg=(90, 60),             # Horizontal and vertical FOV
    u_deg=45,                     # Horizontal rotation
    v_deg=15,                     # Vertical rotation
    out_hw=(720, 1280),           # Output image dimensions
    mode='bilinear'               # Sampling interpolation
)

save_tensor_as_image(perspective_view, "perspective.jpg")
```


## 📚 Basic Functions

### `e2c(e_img, face_w=256, mode='bilinear', cube_format='dice')`
Converts an equirectangular image to a cubemap projection.

- **Parameters**:
  - `e_img` (torch.Tensor): Equirectangular CHW image tensor.
  - `face_w` (int, optional): Cube face width. Default: `256`.
  - `mode` (str, optional): Sampling interpolation mode. Options are `bilinear` and `nearest`. Default: `bilinear`
  - `cube_format` (str, optional): The desired output cubemap format. Options are `dict`, `list`, `horizon`, `stack`, and `dice`. Default: `dice`
    - `stack`: Stack of 6 faces (torch.Tensor), in the order of:  `['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']`.
    - `list`: List of 6 faces (List[torch.Tensor]), in the order of: `['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']`.
    - `dict`: Dictionary with keys pointing to face tensors: (Dict[str, torch.Tensor]). Keys are: `['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']`
    - `dice`: A cubemap in a `dice` layout (`torch.Tensor`).
    - `horizon`: A cubemap in a `horizon` layout (`torch.Tensor`).
  - `channels_first` (bool, optional): Input cubemap channel format (CHW or HWC). Defaults to the PyTorch CHW standard of `True`.

- **Returns**: Cubemap representation of the input image as a tensor, list of tensors, or dict or tensors.

### `c2e(cubemap, h, w, mode='bilinear', cube_format='dice')`
Converts a cubemap projection to an equirectangular image.

- **Parameters**:
  - `cubemap` (torch.Tensor, list of torch.Tensor, or dict of torch.Tensor): Cubemap image tensor, list of tensors, or dict of tensors. Note that tensors should be in the shape of: `CHW`. Inputs should match the corresponding `cube_format`.
  - `h` (int, optional): Output image height. If set to None, `<cube_face_width> * 2` will be used. Default: `None`.
  - `w` (int, optional): Output image width. If set to None, `<cube_face_width> * 4` will be used. Default: `None`.
  - `mode` (str, optional): Sampling interpolation mode. Options are `bilinear` and `nearest`. Default: `bilinear`
  - `cube_format` (str, optional): Input cubemap format. Options are `dict`, `list`, `horizon`, `stack`, and `dice`. Default: `dice`
    - `stack`: Stack of 6 faces (torch.Tensor), in the order of:  `['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']`.
    - `list`: List of 6 faces (List[torch.Tensor]), in the order of: `['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']`.
    - `dict`: Dictionary with keys pointing to face tensors: (Dict[str, torch.Tensor]). Keys are expected to be: `['Front', 'Right', 'Back', 'Left', 'Top', 'Bottom']`
    - `dice`: A cubemap in a `dice` layout (`torch.Tensor`).
    - `horizon`: A cubemap in a `horizon` layout (`torch.Tensor`).
  - `channels_first` (bool, optional): Input cubemap channel format (CHW or HWC). Defaults to the PyTorch CHW standard of `True`.
     
- **Returns**: Equirectangular projection of the input cubemap as a tensor.

### `e2p(e_img, fov_deg, u_deg, v_deg, out_hw, in_rot_deg=0, mode='bilinear')`
Extracts a perspective view from an equirectangular image.

- **Parameters**:
  - `e_img` (torch.Tensor): Equirectangular CHW image tensor.
  - `fov_deg` (float or tuple): Field of view in degrees. If using a tuple, adhere to the following format: (h_fov_deg, v_fov_deg)
  - `u_deg` (float): Horizontal viewing angle in range [-pi, pi]. (- Left / + Right).
  - `v_deg` (float): Vertical viewing angle in range [-pi/2, pi/2]. (- Down/ + Up).
  - `out_hw` (tuple): Output image dimensions in the shape of `(height, width)`.
  - `in_rot_deg` (float, optional): Inplane rotation angle. Default: 0
  - `mode` (str, optional): Sampling interpolation mode. Options are `bilinear` and `nearest`. Default: `bilinear`
  - `channels_first` (bool, optional): Input cubemap channel format (CHW or HWC). Defaults to the PyTorch CHW standard of `True`.

- **Returns**: Perspective view of the equirectangular image as a tensor.


## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.


## 🔬 Citation

If you use this library in your research or project, please refer to the included CITATION.cff file or cite it as follows:

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
