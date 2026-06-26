"""I/O utilities for tracking challenge datasets."""

from dataclasses import dataclass, field
from pathlib import Path

import dask.array as da
import numpy as np
import polars as pl
import torch
import tracksdata as td
import zarr

DEFAULT_SCALE: tuple[float, float, float] = (1.625, 0.40625, 0.40625)


@dataclass
class Dataset:
    path: Path
    image: da.Array | np.ndarray | torch.Tensor | None
    tracks: td.graph.IndexedRXGraph | None
    scale: tuple[float, float, float]
    original_scale: tuple[float, float, float] | None = None
    image_shape: tuple[int, ...] | None = None  # (T, Z, Y, X), always available
    zarr_path: Path | None = None
    quantiles: dict[str, float] = field(default_factory=dict)

    def napari_tracks(self) -> tuple[np.ndarray, dict[int, int]]:
        return td.functional.to_napari_format(
            self.tracks,
            self.image.shape,
            solution_key=None,
            output_tracklet_id_key="track_id",
            mask_key=None,
        )


def open_dataset(ds_path: Path | str,
                target_scale: tuple[float, float, float] | None = None,
                normalize: bool = True,
                gamma: float = 1.0,
                device: str = "cuda",
                require_tracks: bool = False,
                load_image: bool = True,
                downsample: tuple[int, ...] | None = None) -> Dataset:
    """Open a dataset from a zarr file and optionally a geff tracks file.

    Parameters
    ----------
    ds_path : Path or str
        Path to the dataset (without extension, or with .zarr/.geff).
    target_scale : tuple, optional
        Resample to this isotropic (Z, Y, X) voxel scale. If None, no resampling.
    normalize : bool
        Whether to normalize the image intensities.
    gamma : float
        Gamma correction value for normalization.
    device : str
        Device to use for GPU processing.
    require_tracks : bool
        Whether to require a .geff tracks file.
    load_image : bool
        If False, skip loading image data (metadata + tracks only).
    downsample : tuple, optional
        Spatial downsample strides (Z, Y, X). Applied via strided indexing.

    Returns
    -------
    Dataset
        Dataset with image shape (T, Z, Y, X). When ``load_image=False``,
        ``image`` is None but ``image_shape`` and ``zarr_path`` are populated.
    """
    ds_path = Path(ds_path)
    if ds_path.suffix in (".zarr", ".geff"):
        ds_path = ds_path.parent / ds_path.stem

    image_path = ds_path.parent / f"{ds_path.stem}.zarr"
    tracks_path = ds_path.parent / f"{ds_path.stem}.geff"

    if not image_path.exists():
        raise FileNotFoundError(f"Image file not found: {image_path}")
    if require_tracks and not tracks_path.exists():
        raise FileNotFoundError(f"Tracks file not found: {tracks_path}")

    img_ds = zarr.open_group(image_path, mode="r")
    tracks = None
    if require_tracks and tracks_path.exists():
        result = td.graph.IndexedRXGraph.from_geff(tracks_path)
        tracks = result[0] if isinstance(result, tuple) else result

    attrs = dict(img_ds.attrs)
    scale = _parse_scale(attrs)
    quantiles = attrs.get("image_statistics", {}).get("quantiles", {})

    raw_shape = tuple(img_ds["0"].shape)
    if downsample is not None:
        ds_shape = raw_shape[:1] + tuple(-(-s // d) for s, d in zip(raw_shape[1:], downsample))
    else:
        ds_shape = raw_shape

    if not load_image:
        return Dataset(
            path=ds_path, image=None, tracks=tracks, scale=scale,
            image_shape=ds_shape, zarr_path=image_path, quantiles=quantiles,
        )

    da_arr = da.from_zarr(img_ds["0"]).compute()

    resample = target_scale is not None
    original_scale = None
    if resample or normalize:
        if resample:
            original_scale = scale
        da_arr, tracks, scale = _process_on_gpu(
            da_arr, tracks, scale, device,
            resample=resample, target_scale=target_scale,
            normalize=normalize, gamma=gamma,
            precomputed_quantiles=quantiles,
        )

    assert da_arr.ndim == 4

    return Dataset(
        path=ds_path, image=da_arr, tracks=tracks, scale=scale,
        original_scale=original_scale, image_shape=tuple(da_arr.shape),
        zarr_path=image_path, quantiles=quantiles,
    )


def _parse_scale(attrs: dict) -> tuple[float, float, float]:
    """Extract (Z, Y, X) voxel scale from OME-NGFF zarr attrs."""
    if "multiscales" in attrs:
        transform = attrs["multiscales"][0]["datasets"][0]["coordinateTransformations"][0]
        if transform["type"] != "scale":
            raise ValueError(f"Transform type is not 'scale': {transform}")
        return tuple(transform["scale"][-3:])
    return DEFAULT_SCALE


def _lookup_precomputed_quantile(
    quantiles: dict[str, float] | None, q: float, tol: float = 1e-6,
) -> float | None:
    """Return the stored quantile value for *q* if present in *quantiles*."""
    if not quantiles:
        return None
    for key, val in quantiles.items():
        try:
            if abs(float(key) - q) <= tol:
                return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _process_on_gpu(
    image: np.ndarray,
    tracks: td.graph.IndexedRXGraph | None,
    scale: tuple[float, float, float],
    device: str,
    resample: bool = False,
    target_scale: tuple[float, float, float] | None = None,
    normalize: bool = True,
    gamma: float = 1.0,
    q_min: float = 0.000,
    q_max: float = 1.000,
    subsample_factor: int = 50,
    precomputed_quantiles: dict[str, float] | None = None,
) -> tuple[torch.Tensor, td.graph.IndexedRXGraph | None, tuple[float, float, float]]:
    """Process image on GPU: normalization and isotropic resampling.

    All GPU processing is consolidated here to minimize CPU<->GPU transfers.

    Expects 4D input: (T, Z, Y, X)

    Processing order:
    1. Normalize (quantile + gamma) - on CPU/numpy
    2. Transfer to GPU
    3. Resample to isotropic (if enabled)
    4. Transfer back to CPU
    """
    torch_device = torch.device(device)

    # Convert to float32 for processing (zarr data is typically uint16)
    image = image.astype(np.float32)

    # 1. Quantile stats for normalization. Prefer precomputed values from zarr
    #    attrs (instant); fall back to a subsampled np.quantile if missing.
    q1, q2 = None, None
    if normalize:
        q1 = _lookup_precomputed_quantile(precomputed_quantiles, q_min)
        q2 = _lookup_precomputed_quantile(precomputed_quantiles, q_max)
        if q1 is None or q2 is None:
            flat = image.ravel()[::subsample_factor]
            q1, q2 = np.quantile(flat, [q_min, q_max]).astype(np.float32)
        else:
            q1 = np.float32(q1)
            q2 = np.float32(q2)

    # 2. Transfer raw float32 to GPU once (pinned memory for faster DMA)
    tensor = torch.from_numpy(image).pin_memory().to(torch_device, non_blocking=True)

    # 3. Apply normalization on GPU
    if normalize:
        tensor = (tensor - float(q1)) / (float(q2) - float(q1) + 1e-6)
        tensor = tensor.clamp(min=0.0)
        if gamma != 1.0:
            tensor = tensor.pow(gamma)
        tensor = tensor.clamp(0.0, 4.0)

    # 4. Resample to target scale (if enabled)
    if resample:
        scale_arr = np.array(scale)
        if target_scale is None:
            target_scale_val = scale_arr.min()
        else:
            target_scale_val = np.array(target_scale)

        zoom_factors = scale_arr / target_scale_val

        # For 4D input (T, Z, Y, X), reshape to (T, 1, Z, Y, X) for interpolate
        T = tensor.shape[0]
        new_spatial_shape = (np.array(tensor.shape[1:]) * zoom_factors).astype(int).tolist()
        tensor = tensor[:, None]  # (T, 1, Z, Y, X)
        tensor = torch.nn.functional.interpolate(
            tensor, size=new_spatial_shape, mode="trilinear", align_corners=False
        )
        tensor = tensor[:, 0]  # (T, Z, Y, X)

        # Update track coordinates if tracks are provided
        if tracks is not None:
            node_attrs = tracks.node_attrs()
            orig_dtypes = {col: node_attrs.schema[col] for col in ["z", "y", "x"]}
            node_attrs = node_attrs.with_columns(
                (pl.col("z") * zoom_factors[0]).round(0).cast(orig_dtypes["z"]),
                (pl.col("y") * zoom_factors[1]).round(0).cast(orig_dtypes["y"]),
                (pl.col("x") * zoom_factors[2]).round(0).cast(orig_dtypes["x"]),
            )
            tracks.update_node_attrs(
                attrs=node_attrs.select("z", "y", "x").to_dict(),
                node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
            )

        # Update scale to isotropic
        if target_scale is not None:
            scale = target_scale
        else:
            scale = (float(target_scale_val),) * 3

    return tensor, tracks, scale



def invert_time_graph(tracks: td.graph.IndexedRXGraph, max_t: int = 100) -> td.graph.IndexedRXGraph:
    """Invert the time axis of the image and tracks."""

    # Update node time attributes

    node_attrs = tracks.node_attrs()
    node_attrs = node_attrs.with_columns(
        ((max_t - 1) - pl.col("t")).alias("t"),
    )
    tracks.update_node_attrs(
        attrs=node_attrs.select("t").to_dict(),
        node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
    )

    # Collect all edge info first before modifying the graph
    edge_attrs = tracks.edge_attrs()
    edges_to_reverse = []
    for row in edge_attrs.iter_rows(named=True):
        source = row[td.DEFAULT_ATTR_KEYS.EDGE_SOURCE]
        target = row[td.DEFAULT_ATTR_KEYS.EDGE_TARGET]
        edge_id = row[td.DEFAULT_ATTR_KEYS.EDGE_ID]
        attrs = {k: v for k, v in row.items()
                    if k not in (td.DEFAULT_ATTR_KEYS.EDGE_SOURCE,
                                td.DEFAULT_ATTR_KEYS.EDGE_TARGET,
                                td.DEFAULT_ATTR_KEYS.EDGE_ID)}
        edges_to_reverse.append((edge_id, source, target, attrs))

    # Now modify the graph
    for edge_id, source, target, attrs in edges_to_reverse:
        tracks.remove_edge(edge_id=edge_id)
        tracks.add_edge(source_id=target, target_id=source, attrs=attrs)

    return tracks


def rescale_graph_to_original(
    graph: td.graph.BaseGraph,
    original_scale: tuple[float, float, float],
) -> td.graph.BaseGraph:
    """
    Rescale graph coordinates from isotropic space back to original anisotropic space.

    This inverts the isotropic resampling transformation applied during dataset loading.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        Graph with coordinates in isotropic space.
    original_scale : tuple[float, float, float]
        Original (z, y, x) voxel scale before isotropic resampling.

    Returns
    -------
    td.graph.BaseGraph
        Graph with coordinates scaled back to original space.
    """
    scale_arr = np.array(original_scale)
    target_scale = scale_arr.min()
    zoom_factors = scale_arr / target_scale

    # Invert the zoom by dividing coordinates
    inverse_zoom = 1.0 / zoom_factors

    node_attrs = graph.node_attrs()
    node_attrs = node_attrs.with_columns(
        (pl.col("z") * inverse_zoom[0]),
        (pl.col("y") * inverse_zoom[1]),
        (pl.col("x") * inverse_zoom[2]),
    )
    graph.update_node_attrs(
        attrs=node_attrs.select("z", "y", "x").to_dict(),
        node_ids=node_attrs[td.DEFAULT_ATTR_KEYS.NODE_ID].to_list(),
    )

    return graph


def save_graph(graph: td.graph.BaseGraph, output_path: Path | str, overwrite: bool = True) -> None:
    """
    Save a tracksdata graph to a .geff file.

    Parameters
    ----------
    graph : td.graph.BaseGraph
        The graph to save.
    output_path : Path or str
        Output path for the .geff file.
    overwrite : bool
        Whether to overwrite existing files.
    """
    import shutil

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Ensure path has .geff extension
    if not output_path.suffix == ".geff":
        output_path = output_path.with_suffix(".geff")

    # Remove existing file/directory if overwrite is enabled
    if overwrite and output_path.exists():
        if output_path.is_dir():
            shutil.rmtree(output_path)
        else:
            output_path.unlink()

    graph.to_geff(output_path)


def list_datasets(data_dir: Path | str, require_geff: bool = True) -> list[Path]:
    """
    List all valid datasets in a directory.

    A valid dataset has a .zarr file and optionally a corresponding .geff file.

    Parameters
    ----------
    data_dir : Path or str
        Directory containing datasets.
    require_geff : bool
        If True, only return datasets with both .zarr and .geff files.
        If False, return all .zarr files.

    Returns
    -------
    list[Path]
        List of dataset paths (without extension).
    """
    data_dir = Path(data_dir)

    if not data_dir.exists():
        raise FileNotFoundError(f"Directory not found: {data_dir}")

    zarr_files = sorted(data_dir.glob("*.zarr"))

    datasets = []
    for zarr_file in zarr_files:
        if require_geff:
            geff_file = data_dir / f"{zarr_file.stem}.geff"
            if geff_file.exists():
                datasets.append(data_dir / zarr_file.stem)
        else:
            datasets.append(data_dir / zarr_file.stem)

    return datasets
