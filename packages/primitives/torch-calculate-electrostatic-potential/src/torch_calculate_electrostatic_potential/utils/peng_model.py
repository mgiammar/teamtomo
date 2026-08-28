"""Peng 1996 element scattering factor parameters."""

import os

import numpy as np
import torch


def _load_peng_element_scattering_factor_parameter_table():
    path = os.path.join(os.path.dirname(__file__), "peng1996_element_params.npy")
    return np.load(path)


def get_peng_scattering_parameters(
    atomic_numbers: torch.Tensor,
    device: torch.DeviceObjType | None = None,
    dtype: torch.dtype = torch.float32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Look up Peng 1996 Gaussian scattering parameters for atomic numbers.

    Parameters
    ----------
    atomic_numbers : torch.Tensor
        Integer atomic numbers, any shape (...,).
    device : torch.device, optional
        Device to place the output tensors on. If None, defaults to CPU.
    dtype : torch.dtype, optional
        Data type of the output tensors. Defaults to torch.float32.

    Returns
    -------
    a, b : torch.Tensor
        Peng scattering parameters, each shape (..., 5).
    """
    if device is None:
        device = torch.device("cpu")

    table = torch.from_numpy(_load_peng_element_scattering_factor_parameter_table()).to(
        device=device, dtype=dtype
    )
    a, b = table[:, atomic_numbers.to(torch.int64)]
    return a, b
