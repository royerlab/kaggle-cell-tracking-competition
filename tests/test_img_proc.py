import numpy as np
import pytest

from tracking_cellmot.img_proc import (
    nms_3d,
    quantile_normalize,
    rescale_coords_from_isotropic,
    rescale_coords_to_isotropic,
)


def test_quantile_normalize_range():
    """Output should be clipped to [0, clip_max] and have no negatives."""
    rng = np.random.default_rng(0)
    image = rng.integers(0, 65535, size=(10, 16, 16), dtype=np.uint16).astype(np.float32)
    out = quantile_normalize(image, gamma=1.0, clip_max=1.0)
    assert out.min() >= 0.0
    assert out.max() <= 1.0


def test_quantile_normalize_gamma():
    """Higher gamma should darken (compress) the normalized values."""
    rng = np.random.default_rng(1)
    image = rng.random((5, 8, 8)).astype(np.float32)
    out_g1 = quantile_normalize(image, gamma=1.0)
    out_g2 = quantile_normalize(image, gamma=2.0)
    # gamma > 1 squashes values toward 0
    assert out_g2.mean() < out_g1.mean()


def test_quantile_normalize_constant_image():
    """A constant image should not crash (handles near-zero denominator)."""
    image = np.ones((4, 8, 8), dtype=np.float32) * 42.0
    out = quantile_normalize(image)
    assert np.all(np.isfinite(out))


def test_nms_3d_removes_nearby():
    """Points closer than min_distance should be suppressed."""
    coords = np.array([[0, 0, 0], [0, 0, 1], [10, 10, 10]], dtype=np.float32)
    scores = np.array([0.9, 0.5, 0.8])
    scale = (1.0, 1.0, 1.0)
    kept = nms_3d(coords, scores, min_distance=5.0, scale=scale)
    # (0,0,1) is within 5 units of (0,0,0) and has a lower score — should be suppressed
    assert 0 in kept       # highest scorer at (0,0,0)
    assert 1 not in kept   # suppressed
    assert 2 in kept       # far away — kept


def test_nms_3d_empty():
    """Empty input should return empty array."""
    kept = nms_3d(np.empty((0, 3)), np.array([]), min_distance=5.0, scale=(1.0, 1.0, 1.0))
    assert len(kept) == 0


def test_nms_3d_all_far_apart():
    """If all points are far apart, all should be kept."""
    coords = np.array([[0, 0, 0], [100, 0, 0], [0, 100, 0]], dtype=np.float32)
    scores = np.array([0.5, 0.6, 0.7])
    kept = nms_3d(coords, scores, min_distance=5.0, scale=(1.0, 1.0, 1.0))
    assert len(kept) == 3


def test_rescale_coords_roundtrip():
    """Rescaling to isotropic and back should recover the original coordinates."""
    scale = (4.0, 1.0, 1.0)  # anisotropic: Z is 4x coarser
    coords = np.array([[0, 2.0, 5.0, 3.0], [1, 0.0, 1.0, 7.0]], dtype=np.float32)

    iso = rescale_coords_to_isotropic(coords, scale)
    recovered = rescale_coords_from_isotropic(iso, scale)

    np.testing.assert_allclose(recovered, coords, atol=1e-5)


def test_rescale_coords_to_isotropic_z_scaled():
    """Z coordinates should be multiplied by zoom_factor = scale_z / scale_min."""
    scale = (2.0, 1.0, 1.0)
    coords = np.array([[0, 1.0, 1.0, 1.0]], dtype=np.float32)
    iso = rescale_coords_to_isotropic(coords, scale)
    # zoom_factor for Z = 2.0/1.0 = 2 -> z: 1.0 * 2 = 2.0
    assert iso[0, 1] == pytest.approx(2.0)
    assert iso[0, 2] == pytest.approx(1.0)  # Y unchanged
    assert iso[0, 3] == pytest.approx(1.0)  # X unchanged
    assert iso[0, 0] == pytest.approx(0.0)  # T unchanged
