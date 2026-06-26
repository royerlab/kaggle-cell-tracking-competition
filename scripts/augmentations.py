"""Simple image augmentations for training."""

import numpy as np
import torch


def brightness_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
    shift_range: float = 0.1,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random additive brightness shift.

    Parameters
    ----------
    imgs : torch.Tensor
        (W, *spatial) normalised images.
    coords : torch.Tensor
        (W, M, 3) node coordinates — passed through unchanged.
    masks : torch.Tensor
        (W, M) boolean masks — passed through unchanged.
    rng : np.random.Generator
        Random number generator.
    shift_range : float
        Maximum absolute shift sampled from [-shift_range, shift_range].
    """
    shift = rng.uniform(-shift_range, shift_range)
    return imgs + shift, coords, masks


def flip_augment(
    imgs: torch.Tensor,
    coords: torch.Tensor,
    masks: torch.Tensor,
    *,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Random spatial flip: samples uniformly from all 8 axis-aligned symmetries.

    Each of Z, Y, X is independently flipped with probability 0.5,
    giving 2^3 = 8 equally likely outcomes (including identity).

    Creates a copy of coords so the underlying dataset is not mutated.
    Only real (non-padded) coordinates are flipped; padding stays at zero.
    """
    flip_mask = rng.random(3) < 0.5  # (Z, Y, X)
    dims_to_flip = [1 + dim for dim, flip in enumerate(flip_mask) if flip]

    if not dims_to_flip:
        return imgs, coords, masks

    imgs = imgs.flip(dims=dims_to_flip)

    coords = coords.clone()
    shape = imgs.shape[1:]  # (Z, Y, X)
    for dim in range(3):
        if flip_mask[dim]:
            coords[masks, dim] = shape[dim] - coords[masks, dim] - 1

    return imgs, coords, masks
