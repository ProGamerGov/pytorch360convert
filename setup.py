#!/usr/bin/env python3

from setuptools import find_packages, setup

# Load the current project version
exec(open("pytorch360convert/version.py").read())


# Convert relative image links to full links for PyPI
def _relative_to_full_link(long_description: str) -> str:
    """
    Converts relative image links in a README to full GitHub URLs.

    This function replaces relative image links (e.g., in <img> tags and
    Markdown ![]() syntax) with their corresponding full GitHub URLs, appending
    `?raw=true` for direct access to raw images.

    Links are only replaced if they point to the 'examples' directory, and are
    in the format of: `<img src="examples/<image.extension>">` or
    `![](examples/<image.extension>)`.

    Args:
        long_description (str): The text containing relative image links.

    Returns:
        str: The modified text with full image URLs.
    """
    import re

    # Base URL for raw GitHub links
    github_base_url = "https://github.com/ProGamerGov/pytorch360convert/blob/main/"

    # Replace relative links in <img src="examples/...">
    long_description = re.sub(
        r'(<img\s+src="(examples/[\w\-/\.]+)")',
        lambda match: f"{github_base_url}{match.group(2)}?raw=true",
        long_description,
    )

    # Replace relative links in ![](examples/...)
    long_description = re.sub(
        r"(!\[\]\((examples/[\w\-/\.]+)\))",
        lambda match: f"{github_base_url}{match.group(2)}?raw=true",
        long_description,
    )
    return long_description


# Use README.md as the long description
with open("README.md", "r") as fh:
    long_description = fh.read()
long_description = _relative_to_full_link(long_description)

setup(
    name="pytorch360convert",
    version=__version__,  # type: ignore[name-defined]  # noqa: F821
    license="MIT",
    description="360 degree image manipulation and conversion utilities for PyTorch.",
    author="Ben Egan",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/ProGamerGov/pytorch360convert",
    keywords=[
        "360 degree",
        "equirectangular",
        "cubemap",
        "image processing",
        "PyTorch",
        "photo sphere",
        "spherical photo",
        "vr photography",
        "pano",
        "360 photo",
        "360",
        "perspective",
    ],
    python_requires=">=3.7",
    install_requires=[
        "torch>=1.8.0",
    ],
    packages=find_packages(exclude=("tests", "tests.*")),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Processing",
        "Topic :: Software Development",
        "Topic :: Software Development :: Libraries",
        "Topic :: Games/Entertainment",
        "Topic :: Multimedia :: Graphics :: Viewers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Python :: 3.13",
        "Programming Language :: Python :: 3.14",
    ],
)
