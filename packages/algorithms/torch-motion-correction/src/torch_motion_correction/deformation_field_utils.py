"""Utilities for deformation field operations."""

import einops
import torch
import torch.nn.functional as F
from torch_cubic_spline_grids import CubicBSplineGrid3d, CubicCatmullRomGrid3d

from torch_motion_correction.types import DeformationField


def evaluate_deformation_field(
    deformation_field: DeformationField | torch.Tensor,
    tyx: torch.Tensor,  # (..., 3)
    grid_type: str = "catmull_rom",
) -> torch.Tensor:
    """Evaluate shifts from deformation field data.

    Parameters
    ----------
    deformation_field: DeformationField | torch.Tensor
        Deformation field data. Either a
        :class:`~torch_motion_correction.types.DeformationField` (whose ``grid_type``
        is used automatically) or a raw tensor of shape (yx, nt, nh, nw)
        (in which case ``grid_type`` is used).
    tyx: torch.Tensor
        (..., 3) coordinate grid
    grid_type: str
        Type of grid to use ('catmull_rom' or 'bspline'). Ignored when
        ``deformation_field`` is a
        :class:`~torch_motion_correction.types.DeformationField`. Default is
        'catmull_rom'.

    Returns
    -------
    predicted_shifts: torch.Tensor
        (..., 2) predicted shifts
    """
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = DeformationField(
            data=deformation_field, grid_type=grid_type
        )
    return deformation_field.evaluate_at(tyx)


def evaluate_deformation_field_at_t(
    deformation_field: DeformationField
    | torch.Tensor
    | CubicCatmullRomGrid3d
    | CubicBSplineGrid3d,
    t: float,  # [0, 1]
    grid_shape: tuple[int, int],  # (h, w)
    grid_type: str = "catmull_rom",
) -> torch.Tensor:
    """Evaluate a grid of shifts at a specific timepoint from deformation field data.

    Parameters
    ----------
    deformation_field: DeformationField | torch.Tensor | CubicCatmullRomGrid3d | CubicBSplineGrid3d
        Deformation field. When a
        :class:`~torch_motion_correction.types.DeformationField` is passed its
        ``grid_type`` is used automatically. Raw tensors use ``grid_type``. Cubic spline
        grid objects are evaluated directly.
    t: float
        Timepoint to evaluate at [0, 1]
    grid_shape: tuple[int, int]
        (h, w) shape of the grid to evaluate at
    grid_type: str
        Type of grid to use ('catmull_rom' or 'bspline'). Ignored when
        ``deformation_field`` is a
        :class:`~torch_motion_correction.types.DeformationField` or a cubic spline grid
        object. Default is 'catmull_rom'.

    Returns
    -------
    shifts: torch.Tensor
        (2, h, w) predicted shifts
    """
    # Cubic spline grid objects are evaluated directly (legacy path)
    if isinstance(deformation_field, (CubicCatmullRomGrid3d, CubicBSplineGrid3d)):
        device = deformation_field.data.device
        h, w = grid_shape
        y = torch.linspace(0, 1, steps=h, device=device)
        x = torch.linspace(0, 1, steps=w, device=device)
        yy, xx = torch.meshgrid(y, x, indexing="ij")
        yx_grid = einops.rearrange([yy, xx], "yx h w -> (h w) yx")
        tyx_grid = F.pad(yx_grid, (1, 0), value=t)
        shifts = deformation_field(tyx_grid)

        return einops.rearrange(shifts, "(h w) tyx -> tyx h w", h=h, w=w)

    # Wrap raw tensors, then delegate to DeformationField method
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = DeformationField(
            data=deformation_field, grid_type=grid_type
        )

    return deformation_field.evaluate_at_t(t=t, grid_shape=grid_shape)


def resample_deformation_field(
    deformation_field: DeformationField | torch.Tensor,
    target_resolution: tuple[int, int, int],
) -> torch.Tensor:
    """Resample a deformation field to a new resolution.

    Parameters
    ----------
    deformation_field: DeformationField | torch.Tensor
        Deformation field to resample. When a
        :class:`~torch_motion_correction.types.DeformationField` is passed its
        ``grid_type`` is respected during evaluation. Raw tensors use the
        default "catmull_rom" interpolation.
    target_resolution: tuple[int, int, int]
        (nt, nh, nw) target resolution

    Returns
    -------
    torch.Tensor
        Resampled deformation field tensor with shape (2, nt, nh, nw).
    """
    if isinstance(deformation_field, torch.Tensor):
        deformation_field = DeformationField(
            data=deformation_field, grid_type="catmull_rom"
        )

    return deformation_field.resample(target_resolution).data


def image_shifts_to_deformation_field(
    shifts: torch.Tensor,  # (t, 2) shifts in pixels
    pixel_spacing: float,
    device: torch.device = None,
) -> torch.Tensor:
    """Convert whole image shifts to a deformation field for compatibility.

    Parameters
    ----------
    shifts: torch.Tensor
        (t, 2) array of shifts for each frame in pixels (y, x)
    pixel_spacing: float
        Pixel spacing in Angstroms
    device: torch.device, optional
        Device for computation

    Returns
    -------
    deformation_field: torch.Tensor
        (2, t, 1, 1) deformation field with constant shifts per frame
    """
    if device is None:
        device = shifts.device
    else:
        shifts = shifts.to(device)

    # Rescale shifts in pixels to angstroms
    shifts = shifts * pixel_spacing

    # Create deformation field with one yx shift per frame
    deformation_field = einops.rearrange(shifts, "t c -> c t 1 1")

    return deformation_field
