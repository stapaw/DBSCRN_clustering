import math
import time

from rknn import get_nearest_core_point_cluster, set_rknn, set_rknn_ti
from utils import Point


def dbscrn(
    points: list[Point], k: int, m: float = 2, ti: bool = True
) -> dict[str, float]:
    """
    :param points: Input examples.
    :param k: Number of the nearest neighbours. It is assumed that point is in it's k neighbours.
    :param m: Power used in Minkowsky distance function.
    :param ti: If True, uses TI for optimized rk+NN computation.
    """

    start_time = time.perf_counter()
    if ti:
        set_rknn_ti(points, k=k, m=m)
    else:
        set_rknn(points, k=k, m=m)
    rknn_computation_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    s_core = []
    s_non_core = []
    cluster_number = 1

    for point in points:
        if len(point.r_k_plus_nn) < k:
            s_non_core.append(point)
        else:
            s_core.append(point)
            point.point_type = 1

            if point.cluster_id != 0:
                cluster_to_expand = point.cluster_id
            else:
                cluster_to_expand = cluster_number
                cluster_number += 1

            expand_cluster(point, k, cluster_to_expand)
    cluster_expansion_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    for point in s_non_core:
        point.point_type = 0
        if point.cluster_id == 0:
            point.cluster_id = get_nearest_core_point_cluster(point, points, ti)
    non_core_points_cluster_assignment_time = time.perf_counter() - start_time

    return {
        "rk+NN computation runtime": rknn_computation_time,
        "Cluster expansion from core points runtime": cluster_expansion_time,
        "Non-core points cluster assignment runtime": non_core_points_cluster_assignment_time,
    }


def expand_cluster(example: Point, k: int, cluster_number: int) -> None:
    s_tmp = []
    example.cluster_id = cluster_number
    s_tmp.append(example)
    example.visited = True

    for y_k in s_tmp:
        for y_j in y_k.r_k_plus_nn:
            if len(y_j.r_k_plus_nn) > 2 * k / math.pi:
                for p in y_j.r_k_plus_nn:
                    if not p.visited:
                        s_tmp.append(p)
                        p.visited = True
            if y_j.cluster_id == 0:
                y_j.cluster_id = cluster_number
