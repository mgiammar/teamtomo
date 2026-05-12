"""Utilities for extracting image patches around control points."""

import random
from collections.abc import Iterator

import torch


class ImagePatchIterator:
    """Helper data class for iterating over image patches around defined control points.

    NOTE: Patches will be extracted on the same device as the image.

    Attributes
    ----------
    image : torch.Tensor
        The input image (movie frame stack) to draw patches from with shape
        (t, H, W) where t is the number of frames, H is height and W is width.
    image_shape : tuple[int, int, int]
        Shape of the input image (t, H, W).
    patch_size : tuple[int, int]
        Size of the patches to extract (ph, pw) in terms of pixels.
    control_points : torch.Tensor
        Control points in pixel coordinates with shape (t, gh, gw, 3) where
        gh and gw are the number of control points in height and width dimensions,
        and 3 corresponds to (time, y, x) coordinates.
    control_points_normalized : torch.Tensor
        Control points normalized to [0, 1] in all dimensions with shape

    Methods
    -------
    get_iterator(
        batch_size: int, randomized: bool = True
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]
        Data-loader style iterator yielding batches of image patches and corresponding
        normalized control points for each batch.
    """

    image: torch.Tensor  # (t, H, W)
    image_shape: tuple[int, int, int]  # (t, H, W)
    patch_size: tuple[int, int]  # (ph, pw)
    control_points: torch.Tensor  # (t, gh, gw, 3)
    control_points_normalized: torch.Tensor  # (t, gh, gw, 3)

    _points_constant_over_time: bool

    def __init__(
        self,
        image: torch.Tensor,
        patch_size: tuple[int, int],
        control_points: torch.Tensor,
    ) -> None:
        """
        Initialization from image shape, patch size, and control points.

        NOTE: Control points are expected to be in (t, gh, gw, 3) format, and only
        constant control points over time are currently supported.

        Parameters
        ----------
        image : torch.Tensor
            The input image to be patched (t, H, W).
        patch_size : tuple[int, int]
            Size of the patches to extract (ph, pw) in terms of pixels.
        control_points : torch.Tensor
            Control points in pixel coordinates with shape (t, gh, gw, 3) where
            gh and gw are the number of control points in height and width dimensions,
            and 3 corresponds to (time, y, x) coordinates.

        Returns
        -------
        None
        """
        assert len(image.shape) == 3, "Image must be 3D (t, H, W)"
        assert len(patch_size) == 2, "Patch size must be 2D (ph, pw)"
        assert (len(control_points.shape) == 4) and (control_points.shape[-1] == 3), (
            "Control points must be (t, gh, gw, 3)"
        )
        assert image.shape[0] == control_points.shape[0], (
            "Image time dimension and control points time dimension must match"
        )

        self.image = image
        self.image_shape = image.shape
        self.patch_size = patch_size
        self.control_points = control_points.to(image.device)

        # Normalize control points to [0, 1] in all dimensions
        t, H, W = self.image_shape
        self.control_points_normalized = control_points.clone().float()
        self.control_points_normalized[..., 0] /= float(t - 1)
        self.control_points_normalized[..., 1] /= float(H - 1)
        self.control_points_normalized[..., 2] /= float(W - 1)

        # Check if all time slices (zeroth dimension) have the same control point
        # positions in x-y space
        self._points_constant_over_time = torch.all(
            control_points[0, :, :, 1:] == control_points[:, :, :, 1:]
        ).item()

        if not self._points_constant_over_time:
            raise NotImplementedError(
                "Control points varying over time not supported yet"
            )

        # Check that box extraction around extrema won't go out of bounds
        ph, pw = patch_size
        min_y = torch.min(control_points[..., 1]).item()
        max_y = torch.max(control_points[..., 1]).item()
        min_x = torch.min(control_points[..., 2]).item()
        max_x = torch.max(control_points[..., 2]).item()
        err_msg = (
            f"Patch size {patch_size} too large for control points in image "
            f"of shape {self.image_shape} where control points range from "
            f"y: [{min_y}, {max_y}], x: [{min_x}, {max_x}]"
        )
        assert min_y - ph // 2 >= 0, err_msg
        assert max_y + ph // 2 <= H, err_msg
        assert min_x - pw // 2 >= 0, err_msg
        assert max_x + pw // 2 <= W, err_msg

    def get_iterator(
        self, batch_size: int = 1, randomized: bool = True
    ) -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
        """Returns an iterator over image patches and normalized control points.

        Each iteration will yield a stack of image patches with shape
        (batch_size, t, ph, pw) where (ph, pw) is the patch size, and a stack of
        normalized control points with shape (batch_size, t, 3) where 3 corresponds
        to (time, y, x) coordinates in normalized [0, 1] space.

        Parameters
        ----------
        batch_size : int
            Number of patches to return simultaneously. Default is 1.
        randomized : bool
            Whether to randomize the order of patches. Default is True.

        Returns
        -------
        Iterator[tuple[torch.Tensor, torch.Tensor]]
            An iterator yielding tuples of (patches, normalized_control_points)
            where patches is a tensor of shape (batch_size, t, ph, pw) and
            normalized_control_points is a tensor of shape (batch_size, t, 3).
        """

        def inner_iterator() -> Iterator[tuple[torch.Tensor, torch.Tensor]]:
            """Helper function implementing the iterator logic."""
            t, gh, gw, _ = self.control_points.shape
            ph, pw = self.patch_size

            # NOTE: This is currently assuming control points are constant over time
            _control_points = self.control_points[0].reshape(-1, 3)  # (gh * gw, 3)
            # _control_points_norm = self.control_points_normalized[0].reshape(
            #     -1, 3
            # )  # (t, gh * gw, 3)
            _control_points_norm = self.control_points_normalized.reshape(
                t, -1, 3
            )  # (t, gh * gw, 3)

            # Apply randomization if requested
            indices = list(range(gh * gw))
            if randomized:
                random.shuffle(indices)
                _control_points = _control_points[indices]
                _control_points_norm = _control_points_norm[:, indices]

            for i in range(0, gh * gw, batch_size):
                batch_control_points = _control_points[i : i + batch_size]  # (b, 3)
                batch_control_points_norm = _control_points_norm[
                    :, i : i + batch_size
                ]  # (b, t, 3)

                # Use actual control points to extract patches from the image
                patches = []
                for cp in batch_control_points:
                    _, y, x = cp.long()  # NOTE: this will floor float coords...

                    # NOTE: This is assuming no clipping on the boundaries
                    start_y = y - ph // 2
                    end_y = start_y + ph
                    start_x = x - pw // 2
                    end_x = start_x + pw

                    patch = self.image[:, start_y:end_y, start_x:end_x]  # (t, ph, pw)
                    patches.append(patch)

                patches = torch.stack(patches)  # (b, t, ph, pw)

                yield patches, batch_control_points_norm

        return inner_iterator()

    # ── Vectorized precompute methods ─────────────────────────────────────────

    def _extract_all_patches_vectorized(self) -> torch.Tensor:
        """Extract all patches at once using advanced indexing.

        Returns
        -------
        torch.Tensor
            All patches with shape (n_patches, t, ph, pw) on the same device
            as ``self.image``.
        """
        t, gh, gw, _ = self.control_points.shape
        ph, pw = self.patch_size
        device = self.image.device

        flat_cps = self.control_points[0].reshape(-1, 3)  # (n_patches, 3), constant over t
        all_ys = flat_cps[:, 1].long()  # (n_patches,)
        all_xs = flat_cps[:, 2].long()  # (n_patches,)

        y_offsets = torch.arange(-ph // 2, ph - ph // 2, device=device)  # (ph,)
        x_offsets = torch.arange(-pw // 2, pw - pw // 2, device=device)  # (pw,)

        y_indices = all_ys.unsqueeze(1) + y_offsets  # (n_patches, ph)
        x_indices = all_xs.unsqueeze(1) + x_offsets  # (n_patches, pw)

        # image: (t, H, W)
        # y_indices.unsqueeze(-1): (n_patches, ph, 1) — broadcasts over pw
        # x_indices.unsqueeze(-2): (n_patches, 1, pw) — broadcasts over ph
        # result: (t, n_patches, ph, pw)
        all_patches = self.image[
            :,
            y_indices.unsqueeze(-1),
            x_indices.unsqueeze(-2),
        ]
        return all_patches.permute(1, 0, 2, 3).contiguous()  # (n_patches, t, ph, pw)

    def precompute_fft_patches(
        self,
        circle_mask: torch.Tensor,
        store_device: torch.device,
    ) -> torch.Tensor:
        """Precompute masked rFFT for all patches.

        Extraction and FFT are performed on the image's device. The result is
        then transferred to ``store_device`` — typically ``torch.device("cpu")``
        with pin_memory so that per-batch GPU transfers are fast and
        non-blocking. This avoids keeping the full FFT tensor on GPU for large
        movies (e.g. ~17 GB for a (50, 8000, 9000) movie with (2048, 2048)
        patches).

        Parameters
        ----------
        circle_mask : torch.Tensor
            Real-space circular apodisation mask with shape (ph, pw).
        store_device : torch.device
            Device on which to store the precomputed tensor. Pass
            ``torch.device("cpu")`` to use CPU pinned memory.

        Returns
        -------
        torch.Tensor
            (n_patches, t, ph, pw//2+1) complex64 tensor on ``store_device``.
        """
        all_patches = self._extract_all_patches_vectorized()  # (n_patches, t, ph, pw)
        fft = torch.fft.rfftn(all_patches * circle_mask, dim=(-2, -1))
        if store_device.type == "cpu":
            cpu_fft = fft.cpu()
            # pin_memory() accelerates non-blocking transfers to CUDA only
            if torch.cuda.is_available():
                return cpu_fft.pin_memory()
            return cpu_fft
        return fft.to(store_device)

    def get_precomputed_iterator(
        self,
        precomputed_fft: torch.Tensor,
        base_shifts_cache: torch.Tensor,
        batch_size: int = 8,
        randomized: bool = True,
        compute_device: torch.device | None = None,
    ) -> "Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]":
        """Iterator over precomputed FFT patches, centers, and base shifts.

        Each iteration yields a triple of tensors already on ``compute_device``,
        so the forward pass does not need to re-extract patches or re-evaluate
        the frozen base deformation field.

        Parameters
        ----------
        precomputed_fft : torch.Tensor
            (n_patches, t, ph, pw//2+1) complex64 — typically on CPU pinned
            memory, transferred per batch to ``compute_device``.
        base_shifts_cache : torch.Tensor
            (n_patches, t, 2) float32 — frozen base deformation field shifts
            at every patch center, already detached, on ``compute_device``.
        batch_size : int
            Number of patches per mini-batch. Default is 8.
        randomized : bool
            Whether to shuffle patch order each call. Default is True.
        compute_device : torch.device | None
            Device to transfer ``fft_batch`` to. Defaults to the device of
            ``base_shifts_cache``.

        Yields
        ------
        fft_batch : torch.Tensor
            (b, t, ph, pw//2+1) complex — on ``compute_device``.
        centers_batch : torch.Tensor
            (t, b, 3) float — on ``compute_device``.
        base_shifts_batch : torch.Tensor
            (t, b, 2) float — on ``compute_device``.
        """
        if compute_device is None:
            compute_device = base_shifts_cache.device

        def inner_iterator() -> (
            "Iterator[tuple[torch.Tensor, torch.Tensor, torch.Tensor]]"
        ):
            t_dim, gh, gw, _ = self.control_points.shape
            n_patches = gh * gw

            # (t, n_patches, 3) — matches the (t, b, 3) convention of get_iterator
            centers_flat = self.control_points_normalized.reshape(
                t_dim, n_patches, 3
            ).to(compute_device)

            indices = list(range(n_patches))
            if randomized:
                random.shuffle(indices)

            for i in range(0, n_patches, batch_size):
                batch_idx = indices[i : i + batch_size]

                # Transfer from CPU pinned memory → GPU (non-blocking)
                fft_batch = precomputed_fft[batch_idx].to(
                    compute_device, non_blocking=True
                )  # (b, t, ph, pw//2+1)

                centers_batch = centers_flat[:, batch_idx, :]  # (t, b, 3)

                # (n_patches, t, 2)[batch_idx] → (b, t, 2) → (t, b, 2)
                base_shifts_batch = base_shifts_cache[batch_idx].permute(
                    1, 0, 2
                )  # (t, b, 2)

                yield fft_batch, centers_batch, base_shifts_batch

        return inner_iterator()
