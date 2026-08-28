"""AtomStack wraps structure data to the scattering-potential API."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
import torch

from .potential import (
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
)
from .utils.peng_model import get_peng_scattering_parameters

if TYPE_CHECKING:
    from .grid import GridConfig


class AtomStack:
    """Atoms with coordinates, elements, B-factors, and occupancies + helper functions.

    Attributes
    ----------
    device : torch.device
        The device on which the atom data is stored.
    atom_names : list[str] | None
        Optional list of atom names (e.g., ["C", "N", "O"]).
    atomic_numbers : torch.Tensor
        Tensor of integer atomic numbers, shape (N,).
    atom_pos_zyx : torch.Tensor
        Tensor of atom coordinates in Angstroms, shape (..., N, 3).
    atom_bfactors : torch.Tensor
        Tensor of atom B-factors in Angstroms^2, shape (..., N).
    atom_occupancies : torch.Tensor
        Tensor of atom occupancies, shape (..., N).
    atom_params_a : torch.Tensor
        Tensor of Peng scattering amplitude parameters, shape (5, N).
    atom_params_b : torch.Tensor
        Tensor of Peng scattering width parameters, shape (5, N).

    """

    device: torch.device
    atom_names: list[str] | None  # (N,), optional atom names
    atomic_numbers: torch.Tensor  # (N,), integer atomic numbers
    atom_pos_zyx: torch.Tensor  # (..., N, 3), in units of Angstroms
    atom_bfactors: torch.Tensor  # (..., N), B-factors in units of Angstroms^2
    atom_occupancies: torch.Tensor  # (..., N), occupancies, unitless
    atom_params_a: torch.Tensor  # (5, N), amplitude scattering parameters
    atom_params_b: torch.Tensor  # (5, N), width scattering parameters

    def __init__(
        self,
        atom_pos_zyx: torch.Tensor,  # (..., N, 3), Angstroms
        atomic_numbers: torch.Tensor,  # (N,)
        atom_bfactors: torch.Tensor | float = 0.0,  # (..., N) or scalar
        atom_occupancies: torch.Tensor | float = 1.0,  # (..., N) or scalar
        atom_names: list[str] | None = None,
        device: torch.device | str = "cpu",
    ):
        if atom_pos_zyx.ndim < 2 or atom_pos_zyx.shape[-1] != 3:
            raise ValueError(
                "atom_pos_zyx must have shape (..., N, 3), "
                f"got {tuple(atom_pos_zyx.shape)}"
            )
        atomic_numbers = torch.as_tensor(atomic_numbers, dtype=torch.int64)
        if (
            atomic_numbers.ndim != 1
            or atomic_numbers.shape[0] != atom_pos_zyx.shape[-2]
        ):
            raise ValueError(
                f"atomic_numbers must have shape (N,) = ({atom_pos_zyx.shape[-2]},), "
                f"got {tuple(atomic_numbers.shape)}"
            )

        self.device = torch.device(device)
        self.atom_pos_zyx = atom_pos_zyx.to(self.device)
        self.atomic_numbers = atomic_numbers.to(self.device)
        self.atom_names = atom_names

        num_atoms = atom_pos_zyx.shape[-2]
        self.atom_bfactors = self._as_per_atom_tensor(atom_bfactors, num_atoms)
        self.atom_occupancies = self._as_per_atom_tensor(atom_occupancies, num_atoms)

        self.atom_params_a, self.atom_params_b = get_peng_scattering_parameters(
            self.atomic_numbers, device=self.device
        )

    def _as_per_atom_tensor(
        self, value: torch.Tensor | float, num_atoms: int
    ) -> torch.Tensor:
        tensor = torch.as_tensor(value, dtype=torch.float32, device=self.device)
        if tensor.ndim >= 1 and tensor.shape[-1] != num_atoms:
            raise ValueError(
                f"expected trailing dim {num_atoms}, got shape {tuple(tensor.shape)}"
            )
        return tensor

    @property
    def num_atoms(self) -> int:
        """Number of atoms in the stack."""
        return self.atom_pos_zyx.shape[-2]

    def __repr__(self) -> str:
        """Obtain string representation of the AtomStack."""
        batch_shape = tuple(self.atom_pos_zyx.shape[:-2])
        return (
            f"AtomStack(num_atoms={self.num_atoms}, "
            f"batch_shape={batch_shape}, "
            f"device={self.device})"
        )

    @classmethod
    def from_coords_and_names(
        cls,
        atom_pos_zyx: torch.Tensor,
        atom_names: list[str],
        atom_bfactors: torch.Tensor | float = 0.0,
        atom_occupancies: torch.Tensor | float = 1.0,
        device: torch.device | str = "cpu",
    ) -> AtomStack:
        """Construct an AtomStack from coordinates and atom names.

        Parameters
        ----------
        atom_pos_zyx : torch.Tensor
            Atom coordinates in Angstroms, shape (..., N, 3).
        atom_names : list[str]
            Atom names (e.g., ["C", "N", "O"]), length N.
        atom_bfactors : torch.Tensor or float, optional
            Atom B-factors in Angstroms^2, shape (..., N) or scalar, default 0.0.
        atom_occupancies : torch.Tensor or float, optional
            Atom occupancies, shape (..., N) or scalar, default 1.0.
        device : torch.device or str, optional
            Device for the tensors, default "cpu".

        Returns
        -------
        AtomStack
            An AtomStack instance with the provided coordinates and atom names.
        """
        atomic_numbers = torch.tensor(
            [gemmi.Element(name).atomic_number for name in atom_names],
            dtype=torch.int64,
        )
        return cls(
            atom_pos_zyx,
            atomic_numbers,
            atom_bfactors,
            atom_occupancies,
            atom_names,
            device,
        )

    @classmethod
    def from_coords_and_atomic_numbers(
        cls,
        atom_pos_zyx: torch.Tensor,
        atomic_numbers: torch.Tensor,
        atom_bfactors: torch.Tensor | float = 0.0,
        atom_occupancies: torch.Tensor | float = 1.0,
        device: torch.device | str = "cpu",
    ) -> AtomStack:
        """Construct an AtomStack from coordinates and atomic numbers."""
        return cls(
            atom_pos_zyx, atomic_numbers, atom_bfactors, atom_occupancies, None, device
        )

    def to_scattering_potential_3d(
        self, grid_config: GridConfig, **kwargs
    ) -> torch.Tensor:
        """Compute the 3D scattering potential. See `calculate_scattering_potential_3d`.

        Parameters
        ----------
        grid_config : GridConfig
            Grid configuration for the scattering potential volume.
        **kwargs : dict
            Additional keyword arguments to pass to `calculate_scattering_potential_3d`.

        Returns
        -------
        torch.Tensor
            The computed 3D scattering potential volume.
        """
        return calculate_scattering_potential_3d(
            atom_pos_zyx=self.atom_pos_zyx,
            atom_bfactors=self.atom_bfactors,
            atom_params_a=self.atom_params_a,
            atom_params_b=self.atom_params_b,
            grid_config=grid_config,
            atom_occupancies=self.atom_occupancies,
            **kwargs,
        )

    def to_scattering_potential_2d(
        self, grid_config: GridConfig, **kwargs
    ) -> torch.Tensor:
        """Compute the 2D scattering potential. See `calculate_scattering_potential_2d`.

        Parameters
        ----------
        grid_config : GridConfig
            Grid configuration for the scattering potential image.
        **kwargs : dict
            Additional keyword arguments to pass to `calculate_scattering_potential_2d`.

        Returns
        -------
        torch.Tensor
            The computed 2D scattering potential image.
        """
        return calculate_scattering_potential_2d(
            atom_pos_yx=self.atom_pos_zyx[..., 1:],
            atom_bfactors=self.atom_bfactors,
            atom_params_a=self.atom_params_a,
            atom_params_b=self.atom_params_b,
            grid_config=grid_config,
            atom_occupancies=self.atom_occupancies,
            **kwargs,
        )
