# torch-calculate-electrostatic-potential

Differentiable Cryo-EM electrostatic (scattering) potential calculator built with PyTorch. Computes differentiable 2D projected and 3D volumetric potentials from atom coordinates using Peng 1996 scattering factors. Calculations are differentiable with respect to input atomic positions, B-factors, and the values `a`/`b` parameterizing each Gaussian kernel.

## Features

- **`calculate_scattering_potential_3d`** / **`calculate_scattering_potential_2d`**: compute a 3D potential volume or a 2D projected potential image. The 2D case is the exact analytic integral of the 3D potential over the projection axis. Both accept arbitrary leading batch dims (e.g. an ensemble of conformations or a batch of poses).
- **`GridConfig`**: grid geometry — voxel/pixel size and spatial extent (corner points), for either a 3D grid (`grid_shape` length 3) or a 2D grid (length 2). `sublattice_radius` sets the size of the per-atom local region ("stencil") that gets evaluated and scatter-added into the output grid. Increase it when B-factors are high or voxel sizes are small. By default (`equal_length=True`) the grid is symmetrically padded so every axis has the same voxel count as the longest one — a square grid in 2D, a cubic grid in 3D; pass `equal_length=False` to keep an anisotropic voxel count per axis.
- **`AtomStack`**: a thin convenience wrapper — atom positions, elements, B-factors, and occupancies, with a Peng parameter lookup done once at construction. `to_scattering_potential_3d`/`to_scattering_potential_2d` hand its tensors straight to the functions above.

**Axis order**: positions and grids use `(z, y, x)` for 3D and `(y, x)` for 2D — `GridConfig`'s `voxel_size`/corner-point tuples must be given in that same order, since it treats them purely positionally.

## Installation

```sh
# From PyPI (after first release)
pip install torch-calculate-electrostatic-potential
```

```sh
# Development install from GitHub (main branch)
pip install git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
```

With [uv](https://github.com/astral-sh/uv): `uv pip install torch-calculate-electrostatic-potential`.

## Usage

```python
import torch
from torch_calculate_electrostatic_potential import AtomStack, GridConfig

# Atom positions in Angstroms, shape (N, 3) in (z, y, x) order
atom_pos_zyx = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 1.5], [0.0, 1.5, 0.0]])
atom_stack = AtomStack.from_coords_and_names(
    atom_pos_zyx,
    atom_names=["C", "N", "O"],
    atom_bfactors=8 * 3.14159**2 * 0.5**2,  # scalar, per-atom (N,), or batched (..., N)
)

# sublattice_radius is the main complexity lever: larger = more voxels evaluated
# per atom. Scale with B-factors and voxel size (larger radius for larger
# B-factors or finer voxels). Axis order must match atom_pos_zyx: (z, y, x).
grid_3d = GridConfig.from_voxel_size_and_corner_points(
    voxel_size=(1.0, 1.0, 1.0),
    left_bottom_point=(-5.0, -5.0, -5.0),
    right_upper_point=(5.0, 5.0, 5.0),
    sublattice_radius=5.0,
)
potential_volume = atom_stack.to_scattering_potential_3d(grid_3d)
# potential_volume shape: (Dz, Dy, Dx)

grid_2d = GridConfig.from_voxel_size_and_corner_points(
    voxel_size=(1.0, 1.0),
    left_bottom_point=(-5.0, -5.0),
    right_upper_point=(5.0, 5.0),
    sublattice_radius=5.0,
)
potential_image = atom_stack.to_scattering_potential_2d(grid_2d)
# potential_image shape: (Dy, Dx)
```

The lower-level, tensor-only functions this wraps are also public:

```python
from torch_calculate_electrostatic_potential import calculate_scattering_potential_3d, get_peng_scattering_parameters

atom_params_a, atom_params_b = get_peng_scattering_parameters(atom_stack.atomic_numbers)
potential_volume = calculate_scattering_potential_3d(
    atom_pos_zyx, atom_stack.atom_bfactors, atom_params_a, atom_params_b, grid_3d,
)
```

## Testing

Install the package together with test dependencies:

```sh
pip install "torch-calculate-electrostatic-potential[test]" @ git+https://github.com/teamtomo/torch-calculate-electrostatic-potential.git
pytest
```

With coverage: `pytest --cov=torch_calculate_electrostatic_potential --cov-report=html`.

## Requirements

- Python >= 3.9
- PyTorch >= 2.0
- gemmi, numpy, einops, tqdm

## License

BSD 3-Clause License
