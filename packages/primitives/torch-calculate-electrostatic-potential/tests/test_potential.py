import math

import pytest
import torch

from torch_calculate_electrostatic_potential import GridConfig
from torch_calculate_electrostatic_potential.potential import (
    calculate_scattering_potential_2d,
    calculate_scattering_potential_3d,
    evaluate_gaussian_sum,
)


def _manual_point_sample(diff, a, b, bfactor):
    """Reference D-generic point-sample formula, independent of the implementation."""
    D = diff.shape[-1]
    w = 1.0 / (b + bfactor.unsqueeze(-1))
    squared_distance = (diff**2).sum(dim=-1, keepdim=True)
    prefactor = a.unsqueeze(-2) * (4 * math.pi) ** (D / 2) * w.unsqueeze(-2) ** (D / 2)
    exponent = torch.exp(-4 * math.pi**2 * squared_distance * w.unsqueeze(-2))
    return (prefactor * exponent).sum(dim=-1)


class TestEvaluateGaussianSum:
    @pytest.mark.parametrize("D", [2, 3])
    def test_point_sample_matches_manual_formula(self, D):
        torch.manual_seed(0)
        N, K = 4, 5
        a = torch.rand(N, 5) * 2 + 0.1
        b = torch.rand(N, 5) * 3 + 0.5
        bfactor = torch.rand(N) * 10
        voxel_size = torch.rand(D) * 0.5 + 0.5
        diff = torch.randn(N, K, D) * 2

        got = evaluate_gaussian_sum(
            diff, a, b, bfactor, voxel_size, per_voxel_averaging=False
        )
        expected = _manual_point_sample(diff, a, b, bfactor)
        assert torch.allclose(got, expected, atol=1e-6)

    @pytest.mark.parametrize("D", [2, 3])
    def test_voxel_averaged_converges_to_point_sample_as_voxel_shrinks(self, D):
        torch.manual_seed(1)
        N, K = 3, 4
        a = torch.rand(N, 5) * 2 + 0.1
        b = torch.rand(N, 5) * 3 + 0.5
        bfactor = torch.rand(N) * 10
        diff = torch.randn(N, K, D) * 2

        # Small enough to demonstrate convergence, not so small that the erf-difference
        # formula loses precision to float32 cancellation.
        small_voxel = torch.full((D,), 1e-2)
        averaged = evaluate_gaussian_sum(
            diff, a, b, bfactor, small_voxel, per_voxel_averaging=True
        )
        point = evaluate_gaussian_sum(
            diff, a, b, bfactor, small_voxel, per_voxel_averaging=False
        )
        assert torch.allclose(averaged, point, rtol=1e-3, atol=1e-3)

    def test_2d_matches_z_integrated_3d(self):
        """The core insight: D=2 is the exact z-integral of D=3, not a sum over it."""
        a = torch.tensor([[1.3]])
        b = torch.tensor([[2.1]])
        bfactor = torch.tensor([5.0])
        xy = torch.tensor([[0.3, -0.4], [0.0, 0.0], [1.5, 0.7]]).unsqueeze(0)

        direct_2d = evaluate_gaussian_sum(
            xy, a, b, bfactor, torch.tensor([1.0, 1.0]), per_voxel_averaging=False
        )

        z = torch.linspace(-200.0, 200.0, 400_001, dtype=torch.float64)
        results = []
        for x, y in xy[0]:
            diff_z = torch.stack(
                [x.double().expand_as(z), y.double().expand_as(z), z], dim=-1
            ).unsqueeze(0)
            vals = evaluate_gaussian_sum(
                diff_z,
                a.double(),
                b.double(),
                bfactor.double(),
                torch.tensor([1.0, 1.0, 1.0], dtype=torch.float64),
                per_voxel_averaging=False,
            )
            results.append(torch.trapz(vals.squeeze(0), z))
        integrated = torch.stack(results).float()

        assert torch.allclose(direct_2d.squeeze(0), integrated, rtol=1e-4)


class TestCalculateScatteringPotential:
    def _grid(self, ndim, sublattice_radius=5.0):
        voxel_size = (1.0,) * ndim
        lo = (-5.0,) * ndim
        hi = (5.0,) * ndim
        return GridConfig.from_voxel_size_and_corner_points(
            voxel_size=voxel_size,
            left_bottom_point=lo,
            right_upper_point=hi,
            sublattice_radius=sublattice_radius,
        )

    def test_3d_shape(self, simple_atoms):
        grid = self._grid(3)
        vol = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
        )
        assert vol.shape == tuple(grid.grid_shape.tolist())
        assert torch.isfinite(vol).all()
        assert vol.min() >= 0

    def test_2d_shape(self, simple_atoms):
        grid = self._grid(2)
        img = calculate_scattering_potential_2d(
            simple_atoms["atom_pos_zyx"][:, 1:],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
        )
        assert img.shape == tuple(grid.grid_shape.tolist())
        assert torch.isfinite(img).all()

    def test_rejects_wrong_grid_ndim(self, simple_atoms):
        grid_2d = self._grid(2)
        with pytest.raises(ValueError):
            calculate_scattering_potential_3d(
                simple_atoms["atom_pos_zyx"],
                simple_atoms["atom_bfactors"],
                simple_atoms["atom_params_a"],
                simple_atoms["atom_params_b"],
                grid_2d,
            )

    def test_batched_leading_dims_match_unbatched(self, simple_atoms):
        grid = self._grid(3)
        single = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
        )
        batched_pos = simple_atoms["atom_pos_zyx"].unsqueeze(0).repeat(4, 1, 1)
        batched = calculate_scattering_potential_3d(
            batched_pos,
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
        )
        assert batched.shape == (4, *single.shape)
        for i in range(4):
            assert torch.allclose(batched[i], single)

    def test_atom_batch_size_does_not_change_result(self, simple_atoms):
        grid = self._grid(3)
        full = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
            batch_size=4096,
        )
        chunked = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
            batch_size=1,
        )
        assert torch.allclose(full, chunked, atol=1e-6)

    def test_zero_occupancy_atom_contributes_nothing(self, simple_atoms):
        grid = self._grid(3)
        occ_all = torch.ones(3)
        occ_drop_one = torch.tensor([1.0, 1.0, 0.0])
        with_all = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
            atom_occupancies=occ_all,
        )
        without_third = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"][:2],
            simple_atoms["atom_bfactors"][:2],
            simple_atoms["atom_params_a"][:2],
            simple_atoms["atom_params_b"][:2],
            grid,
        )
        dropped = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
            atom_occupancies=occ_drop_one,
        )
        assert not torch.allclose(with_all, dropped)
        assert torch.allclose(dropped, without_third, atol=1e-6)

    def test_broadcast_unbatched_params_against_batched_positions(self, simple_atoms):
        grid = self._grid(3)
        pos_batched = simple_atoms["atom_pos_zyx"].unsqueeze(0).repeat(3, 1, 1)
        # a, b, bfactor all unbatched (N, ...) while positions are batched (B, N, ...)
        out = calculate_scattering_potential_3d(
            pos_batched,
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
        )
        assert out.shape == (3, *grid.grid_shape.tolist())


class TestGradients:
    """Differentiability is the headline requirement of this API."""

    def _grid(self):
        return GridConfig.from_voxel_size_and_corner_points(
            voxel_size=(1.0, 1.0, 1.0),
            left_bottom_point=(-5.0, -5.0, -5.0),
            right_upper_point=(5.0, 5.0, 5.0),
            sublattice_radius=5.0,
        )

    @pytest.mark.parametrize(
        "input_name",
        ["atom_pos_zyx", "atom_bfactors", "atom_params_a", "atom_params_b"],
    )
    def test_gradient_flows_to_input(self, simple_atoms, input_name):
        grid = self._grid()
        inputs = {k: v.clone() for k, v in simple_atoms.items()}
        inputs[input_name].requires_grad_(True)

        out = calculate_scattering_potential_3d(
            inputs["atom_pos_zyx"],
            inputs["atom_bfactors"],
            inputs["atom_params_a"],
            inputs["atom_params_b"],
            grid,
        )
        out.sum().backward()
        grad = inputs[input_name].grad
        assert grad is not None
        assert torch.isfinite(grad).all()
        assert grad.abs().sum() > 0

    def test_gradient_flows_to_occupancies(self, simple_atoms):
        grid = self._grid()
        occ = torch.rand(3, requires_grad=True)
        out = calculate_scattering_potential_3d(
            simple_atoms["atom_pos_zyx"],
            simple_atoms["atom_bfactors"],
            simple_atoms["atom_params_a"],
            simple_atoms["atom_params_b"],
            grid,
            atom_occupancies=occ,
        )
        out.sum().backward()
        assert (
            occ.grad is not None
            and torch.isfinite(occ.grad).all()
            and occ.grad.abs().sum() > 0
        )

    def test_position_gradient_matches_central_difference(self, simple_atoms):
        """Numerically verify d(sum of potential)/d(atom position) via central diff."""
        grid = self._grid()

        def loss_fn(pos):
            vol = calculate_scattering_potential_3d(
                pos,
                simple_atoms["atom_bfactors"],
                simple_atoms["atom_params_a"],
                simple_atoms["atom_params_b"],
                grid,
            )
            return vol.sum()

        pos = simple_atoms["atom_pos_zyx"].clone().requires_grad_(True)
        loss_fn(pos).backward()
        analytical = pos.grad[0, 0].item()  # d(loss)/d(z of atom 0)

        eps = 1e-3
        pos_plus = simple_atoms["atom_pos_zyx"].clone()
        pos_plus[0, 0] += eps
        pos_minus = simple_atoms["atom_pos_zyx"].clone()
        pos_minus[0, 0] -= eps
        numerical = (loss_fn(pos_plus).item() - loss_fn(pos_minus).item()) / (2 * eps)

        assert analytical == pytest.approx(numerical, rel=1e-2, abs=1e-3)
