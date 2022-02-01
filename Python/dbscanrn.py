import time
from heapq import nsmallest
from typing import Dict, List, Optional, Tuple

from dbscan import assign_clusters_dbscan
from tqdm import tqdm
from utils import Point, distance_fn_generator


def dbscanrn(
    points: List[Point],
    k: int,
    m: float = 2,
    ti: bool = True,
    ref_point: Optional[Point] = None,
) -> Dict[str, float]:
    """
    :param points: Input examples.
    :param k: Number of the nearest neighbours. It is assumed that point is in it's k neighbours.
    :param m: Power used in Minkowsky distance function.
    :param ti: If True, uses TI for optimized rk+NN computation.
    :param ref_point: Reference point used by TI optimized version.
    """

    if ti:
        point_distance_time, rknn_time = set_rknn_ti(
            points=points, ref_point=ref_point, m=m, k=k
        )
    else:
        point_distance_time, rknn_time = set_rknn(points=points, m=m, k=k)

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
        "2_sort_by_ref_point_distances": point_distance_time,
        "3_eps_neighborhood/rnn_calculation": rknn_time,
        "4_clustering": clustering_time,
    }


def compute_point_idx_ref_distance_list(
    points: List[Point], ref_point: Point, m_power: float
) -> Tuple[float, List[Tuple[int, float]]]:
    start_time = time.perf_counter()

    dist_fn = distance_fn_generator(m_power)
    point_idx_ref_dist = sorted(
        [
            (i, dist_fn(p, ref_point))
            for i, p in tqdm(
                enumerate(points), desc="Calculating reference distances..."
            )
        ],
        key=lambda pair: pair[1],
    )
    point_distance_time = time.perf_counter() - start_time

    return point_distance_time, point_idx_ref_dist


def set_rknn(
    points: List[Point],
    k: int,
    m: float,
    k_plus_nn_tolerance: float = 10e-9,
) -> Tuple[float, float]:
    start_time = time.perf_counter()

    dist_fn = distance_fn_generator(m)
    for i in tqdm(range(len(points)), desc="Calculating rK+NN..."):
        point_idx_to_dist = [
            (j, dist_fn(points[i], points[j])) for j in range(len(points)) if j != i
        ]

        k_plus_nn = nsmallest(
            k - 1, iterable=point_idx_to_dist, key=lambda pair: pair[1]
        )  # point is in it's rKNN

        point_idx_to_dist = [
            idx_dist for idx_dist in point_idx_to_dist if idx_dist not in k_plus_nn
        ]
        k_plus_nn_candidate = nsmallest(1, point_idx_to_dist, key=lambda pair: pair[1])[
            0
        ]
        while abs(k_plus_nn_candidate[1] - k_plus_nn[-1][1]) < k_plus_nn_tolerance:
            k_plus_nn.append(k_plus_nn_candidate)
            point_idx_to_dist = [
                idx_dist for idx_dist in point_idx_to_dist if idx_dist not in k_plus_nn
            ]
            k_plus_nn_candidate = nsmallest(
                1, point_idx_to_dist, key=lambda pair: pair[1]
            )[0]
        k_plus_nn_indices = [neighbour_idx for (neighbour_idx, _) in k_plus_nn]

        for idx in k_plus_nn_indices:
            if points[idx].r_k_plus_nn is None:
                points[idx].r_k_plus_nn = [points[i]]
            else:
                points[idx].r_k_plus_nn.append(points[i])

        points[i].k_plus_nn = [points[i]] + [points[idx] for idx in k_plus_nn_indices]

    for point in points:
        if point.r_k_plus_nn is None:
            point.r_k_plus_nn = []
    rknn_time = time.perf_counter() - start_time
    return 0, rknn_time


def set_rknn_ti(
    points: List[Point],
    ref_point: Point,
    k: int,
    m: float = 2.0,
    k_plus_nn_tolerance: float = 10e-9,
) -> Tuple[float, float]:
    (
        point_distance_time,
        point_idx_ref_dist,
    ) = compute_point_idx_ref_distance_list(points, ref_point, m)

    start_time = time.perf_counter()
    for i in tqdm(range(len(points)), desc="Calculating rK+NN using TI..."):
        current_point_idx, current_point_ref_dist = point_idx_ref_dist[i]
        current_point = points[current_point_idx]

        dist_fn = distance_fn_generator(m)
        _, current_point_ref_dist = point_idx_ref_dist[i]

        prev_idx_diff = 1
        next_idx_diff = 1
        search_prev = (i - prev_idx_diff) >= 0
        search_next = (i + next_idx_diff) <= (len(points) - 1)

        candidate_point_real_dist: List[Tuple[int, float]] = []
        eps = 0.0
        stop_search = False
        k_corrected = k - 1  # account for point being it's own kNN

        while len(candidate_point_real_dist) < k_corrected or not stop_search:
            if len(candidate_point_real_dist) == k_corrected:
                eps = max(candidate_point_real_dist, key=lambda pair: pair[1])[1]
                current_point.max_eps = eps

            if not search_prev and search_next:
                go_next = True
                pessimistic_estimation = (
                    point_idx_ref_dist[i + next_idx_diff][1] - current_point_ref_dist
                )

            elif search_prev and not search_next:
                go_next = False
                pessimistic_estimation = (
                    current_point_ref_dist - point_idx_ref_dist[i - prev_idx_diff][1]
                )

            elif search_prev and search_next:
                ref_dist_diff_next = (
                    point_idx_ref_dist[i + next_idx_diff][1] - current_point_ref_dist
                )
                ref_dist_diff_prev = (
                    current_point_ref_dist - point_idx_ref_dist[i - prev_idx_diff][1]
                )
                if ref_dist_diff_next < ref_dist_diff_prev:
                    go_next = True
                    pessimistic_estimation = ref_dist_diff_next
                else:
                    go_next = False
                    pessimistic_estimation = ref_dist_diff_prev
            else:
                stop_search = True
                break

            if go_next:
                current_ref_dist_idx = i + next_idx_diff
            else:
                current_ref_dist_idx = i - prev_idx_diff

            if (
                len(candidate_point_real_dist) >= k_corrected
                and pessimistic_estimation > eps
            ):
                stop_search = True

            else:
                current_candidate_idx = point_idx_ref_dist[current_ref_dist_idx][0]
                current_candidate_point = points[current_candidate_idx]
                current_point_real_dist = dist_fn(
                    current_point, current_candidate_point
                )

                if len(candidate_point_real_dist) < k_corrected:
                    candidate_point_real_dist.append(
                        (current_candidate_idx, current_point_real_dist)
                    )
                else:
                    # Account for floating point errors
                    if (
                        current_point_real_dist < eps
                        or abs(current_point_real_dist - eps) <= k_plus_nn_tolerance
                    ):
                        candidate_point_real_dist.append(
                            (current_candidate_idx, current_point_real_dist)
                        )
                        eps = nsmallest(
                            k_corrected,
                            [dist for _, dist in candidate_point_real_dist],
                        )[-1]
                if go_next:
                    next_idx_diff += 1
                    search_next = (i + next_idx_diff) <= (len(points) - 1)
                else:
                    prev_idx_diff += 1
                    search_prev = (i - prev_idx_diff) >= 0

        # Determine final k+NN
        if len(candidate_point_real_dist) > k_corrected:
            sorted_candidate_idx_to_real_dist = sorted(
                candidate_point_real_dist, key=lambda pair: pair[1]
            )
            k_plus_nn_idx_dist = sorted_candidate_idx_to_real_dist[:k_corrected]
            for (
                candidate_idx,
                candidate_dist,
            ) in sorted_candidate_idx_to_real_dist[k_corrected:]:
                if (
                    abs(k_plus_nn_idx_dist[-1][1] - candidate_dist)
                    <= k_plus_nn_tolerance
                ):
                    k_plus_nn_idx_dist.append((candidate_idx, candidate_dist))
                else:
                    break
        else:
            k_plus_nn_idx_dist = candidate_point_real_dist

        current_point.min_eps = eps

        k_plus_nn = [points[neighbour_idx] for (neighbour_idx, _) in k_plus_nn_idx_dist]
        for neighbour in k_plus_nn:
            if neighbour.r_k_plus_nn is None:
                neighbour.r_k_plus_nn = [current_point]
            else:
                neighbour.r_k_plus_nn.append(current_point)

        current_point.k_plus_nn = [current_point] + k_plus_nn

    for point in points:
        if point.r_k_plus_nn is None:
            point.r_k_plus_nn = []
    rknn_time = time.perf_counter() - start_time

    return point_distance_time, rknn_time
