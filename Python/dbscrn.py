import math
import time

from dbscan import assign_clusters_dbscan
from rknn import set_rknn, set_rknn_ti
from tqdm import tqdm
from utils import Point, distance_fn_generator, get_pairwise_distances


def dbscrn(
    points: list[Point], k: int, m: float = 2, ti: bool = True
) -> dict[str, float]:
    """
    :param points: Input examples.
    :param k: Number of the nearest neighbours. It is assumed that point is in it's k neighbours.
    :param m: Power used in Minkowsky distance function.
    :param ti: If True, uses TI for optimized rk+NN computation.
    """

    if ti:
        start_time = time.perf_counter()

        ref_point = Point(id=-1, vals=[0.0 for _ in points[0].vals])
        dist_fn = distance_fn_generator(m)

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

        start_time = time.perf_counter()
        set_rknn_ti(points=points, point_idx_ref_dist=point_idx_ref_dist, k=k, m=m)
        rknn_time = time.perf_counter() - start_time
    else:
        start_time = time.perf_counter()
        pairwise_distances = get_pairwise_distances(points, m)
        point_distance_time = time.perf_counter() - start_time

        start_time = time.perf_counter()
        set_rknn(points=points, pairwise_distances=pairwise_distances, k=k)
        rknn_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    core_points = [p for p in points if len(p.r_k_plus_nn) >= k]
    non_core_points = [p for p in points if len(p.r_k_plus_nn) < k]
    for core_point in core_points:
        core_point.point_type = 1

    assign_clusters_dbscan(
        core_points=core_points,
        non_core_points=non_core_points,
        neighbours_getter=lambda p: p.r_k_plus_nn,
    )
    clustering_time = time.perf_counter() - start_time

    return {
        "2_sort_by_ref_point_distances": point_distance_time,
        "3_eps_neighborhood/rnn_calculation": rknn_time,
        "4_clustering": clustering_time,
    }
