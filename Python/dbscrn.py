import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from dbscan import assign_clusters_dbscan
from rknn import set_rknn, set_rknn_ti
from tqdm import tqdm
from utils import Point, distance_fn_generator


def dbscrn(
    points: List[Point],
    k: int,
    m: float = 2,
    ti: bool = True,
    pairwise_distances: Optional[Dict[Tuple[int, int], float]] = None,
    point_idx_ref_dist: List[Tuple[int, float]] = None,
) -> Dict[str, float]:
    """
    :param points: Input examples.
    :param k: Number of the nearest neighbours. It is assumed that point is in it's k neighbours.
    :param m: Power used in Minkowsky distance function.
    :param ti: If True, uses TI for optimized rk+NN computation.
    :param pairwise_distances: Precomputed pairwise point distances for non-optimized version
    of algorithm.
    :param point_idx_ref_dist: Precomputed tuples containing point indices and point distances to
    the reference point.
    """

    if ti:
        start_time = time.perf_counter()
        set_rknn_ti(points=points, point_idx_ref_dist=point_idx_ref_dist, k=k, m=m)
        rknn_time = time.perf_counter() - start_time
    else:
        start_time = time.perf_counter()
        set_rknn(points=points, pairwise_distances=pairwise_distances, k=k)
        rknn_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    core_points = [p for p in points if len(p.r_k_plus_nn) >= k]
    non_core_points = [p for p in points if len(p.r_k_plus_nn) < k]

    assign_clusters_dbscan(
        core_points=core_points,
        non_core_points=non_core_points,
        neighbours_getter_cp=lambda p: p.r_k_plus_nn,
        neighbours_getter_ncp=lambda p: p.k_plus_nn,
    )
    clustering_time = time.perf_counter() - start_time

    return {
        "3_eps_neighborhood/rnn_calculation": rknn_time,
        "4_clustering": clustering_time,
    }


def compute_point_idx_ref_distance_list(
    points: List[Point], m_power: float, cache: Optional[Path] = None
) -> Tuple[float, Point, List[Tuple[int, float]]]:
    start_time = time.perf_counter()
    ref_point = Point(
        id=-1,
        vals=[min(p.vals[i] for p in points) for i in range(len(points[0].vals))],
    )
    if cache is not None and cache.exists():
        with cache.open("r") as f:
            point_idx_ref_dist = json.load(f)

    if cache is None or not cache.exists():
        dist_fn = distance_fn_generator(m_power)
        point_idx_ref_dist = sorted(
            [
                (i, dist_fn(ref_point, p))
                for i, p in tqdm(
                    enumerate(points), desc="Calculating reference distances..."
                )
            ],
            key=lambda pair: pair[1],
        )
    point_distance_time = time.perf_counter() - start_time
    if cache is not None and not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        with cache.open("w+") as f:
            json.dump([(idx, round(dist, 3)) for (idx, dist) in point_idx_ref_dist], f, indent=2)

    return point_distance_time, ref_point, point_idx_ref_dist
