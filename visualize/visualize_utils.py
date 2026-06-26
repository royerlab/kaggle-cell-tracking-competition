import napari
import numpy as np


def _compute_contrast_limits(array: np.ndarray, q_low: float = 0.01, q_high: float = 0.999, subsample: int = 10) -> tuple[float, float]:
    """
    Compute contrast limits using quantiles on a subsampled image.
    
    Parameters
    ----------
    array : np.ndarray
        Input array.
    q_low : float
        Lower quantile (default 0.01 = 1st percentile).
    q_high : float
        Upper quantile (default 0.999 = 99.9th percentile).
    subsample : int
        Subsample factor for efficiency (default 10 = use every 10th element).
    
    Returns
    -------
    tuple[float, float]
        (vmin, vmax) contrast limits.
    """
    # Subsample for efficiency
    flat = array.ravel()[::subsample]
    vmin = float(np.quantile(flat, q_low))
    vmax = float(np.quantile(flat, q_high))
    return (vmin, vmax)


def show_with_napari(*arrays,
                    graph = None,
                    scale: tuple[float, float, float] = (1, 1, 1),
                    names = None,
                    ndisplay: int = 3,
                    title: str | None = None,
                    show_next_button: bool = False) -> None:
    """
    Show a set of arrays and a graph in napari.
    
    Contrast limits are automatically computed using quantile normalization
    on a subsampled version of each image for efficiency.
    
    e.g. show_with_napari(image, target, graph=ds.napari_tracks(), scale=ds.scale, names=["image", "target"])
    
    Args:
        *arrays: Arrays to show.
        graph: Graph to show.
        scale: Scale of the arrays.
        names: Names of the arrays.
        ndisplay: Number of dimensions to display (default 3 for 3D view).
    """
    if names is None:
        names = [f"array_{i}" for i in range(len(arrays))]

    assert len(arrays) == len(names), "Number of arrays and names must match"

    viewer = napari.Viewer(ndisplay=ndisplay, title=title or "napari")

    if len(arrays) == 1:
        colormap = ['gray']
    else:
        colormap = ['green', 'red', 'blue', 'yellow', 'purple', 'orange', 'pink', 'brown', 'gray', 'black']
    
    for i, array in enumerate(arrays):
        # Convert to numpy if needed
        if hasattr(array, 'numpy'):
            if hasattr(array, 'cpu'):
                array = array.cpu().numpy()
            else:
                array = array.numpy()
        elif hasattr(array, 'compute'):
            array = array.compute()
        
        # Compute contrast limits using quantile on subsampled data
        contrast_limits = _compute_contrast_limits(array)
        
        viewer.add_image(
            array, 
            scale=scale, 
            contrast_limits=contrast_limits, 
            name=names[i], 
            blending='additive', 
            colormap=colormap[i % len(colormap)]
        )
    
    if graph is not None:
        tracklets, tracklets_graph = graph
        viewer.add_tracks(
            tracklets,
            graph=tracklets_graph,
            scale=scale,
            tail_width=5,
            tail_length=100,
            opacity=1.0,
        )

    if show_next_button:
        from qtpy.QtWidgets import QPushButton
        btn = QPushButton("Next Dataset ▶")
        btn.clicked.connect(viewer.close)
        viewer.window.add_dock_widget(btn, area="left", name="Controls")

    napari.run()