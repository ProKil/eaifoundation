import contextlib
import time
import warnings
from typing import Any, Dict

import numpy as np
import numpy.typing as npt
from allenact_plugins.ithor_plugin.ithor_environment import (
    IThorEnvironment,
)


@contextlib.contextmanager
def timer():
    start = time.time()
    try:
        yield None
    finally:
        print(f"Ran for {time.time() - start:2g} seconds")


def get_object_radius(obj: Dict[str, Any]):
    if obj["objectOrientedBoundingBox"] is not None:
        corners = np.array(obj["objectOrientedBoundingBox"]["cornerPoints"])
    else:
        corners = np.array(obj["axisAlignedBoundingBox"]["cornerPoints"])

    a = corners - corners.mean(0, keepdims=True)
    return float(np.sqrt((a * a).sum(-1)).max())


def position_to_floor_tuple(p: Dict[str, float]):
    return (p["x"], p["z"])


def position_to_numpy(p: Dict[str, float], with_y: bool = True):
    if with_y:
        return np.array([p["x"], p["y"], p["z"]])
    else:
        return np.array([p["x"], p["z"]])


def sigmoid(x: float):
    return 1.0 / (1 + np.exp(-x))


def sigmoid_margin_distance(dist: float, margin: float, scale: float):
    return max(
        2 * (sigmoid(scale * max(dist - margin, 0)) - 0.5),
        0,
    )


def absolute_margin_distance(dist: float, margin: float, scale: float):
    return scale * max(dist - margin, 0)


def get_angle_of_vec_counter_clockwise(vec: npt.NDArray):
    norm = np.linalg.norm(vec)
    if norm > 1e-8:
        vec = vec / norm

        if abs(vec[0]) < 1e-6:
            return (np.pi / 2.0) * (1 if vec[1] > 0 else -1)
        else:
            return np.arctan(vec[1] / vec[0]) + (np.pi if vec[0] < 0 else 0)
    else:
        warnings.warn(
            f"norm of vector {vec} was too small, returning an angle of 0"
        )
        return 0.0


def fast_dist_between_objects_lower_bound(
    o0: Dict[str, Any], o1: Dict[str, Any]
):
    return max(
        IThorEnvironment.position_dist(o0["position"], o1["position"])
        - o0["radius"]
        - o1["radius"],
        0,
    )


def fast_dist_to_object_lower_bound(
    from_pos: Dict[str, Any], obj: Dict[str, Any]
):
    return max(
        IThorEnvironment.position_dist(from_pos, obj["position"])
        - obj["radius"],
        0,
    )
