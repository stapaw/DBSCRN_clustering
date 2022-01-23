from heapq import nsmallest
from typing import List, Tuple

from tqdm import tqdm
from utils import Point, distance_fn_generator


def set_rknn(
    points: List[Point],
    pairwise_distances: dict[tuple[int, int], float],
    k: int,
    k_plus_nn_tolerance: float = 10e-9,
) -> None:
    for i in tqdm(range(len(points)), desc="Calculating rK+NN..."):
        point_idx_to_dist = [
            (m if m != i else n, pairwise_distances[(m, n) if m < n else (n, m)])
            for m in range(len(points))
            for n in range(m + 1, len(points))
            if i in (m, n)
        ]

        k_plus_nn_indices = get_knn_indices(point_idx_to_dist, k, k_plus_nn_tolerance)

        points[i].k_plus_nn = [points[idx] for idx in k_plus_nn_indices]
        for idx in k_plus_nn_indices:
            if points[idx].r_k_plus_nn is None:
                points[idx].r_k_plus_nn = [points[i]]
            else:
                points[idx].r_k_plus_nn.append(points[i])

    for point in points:
        if point.r_k_plus_nn is None:
            point.r_k_plus_nn = []


def get_knn_indices(
    point_idx_to_dist: list[tuple[int, float]],
    k: int,
    k_plus_nn_tolerance: float = 10e-9,
) -> list[int]:
    k_plus_nn = nsmallest(
        k - 1, iterable=point_idx_to_dist, key=lambda pair: pair[1]
    )  # point is in it's rKNN

    point_idx_to_dist = [
        idx_dist for idx_dist in point_idx_to_dist if idx_dist not in k_plus_nn
    ]
    k_plus_nn_candidate = nsmallest(1, point_idx_to_dist, key=lambda pair: pair[1])[0]
    while abs(k_plus_nn_candidate[1] - k_plus_nn[-1][1]) < k_plus_nn_tolerance:
        k_plus_nn.append(k_plus_nn_candidate)
        point_idx_to_dist = [
            idx_dist for idx_dist in point_idx_to_dist if idx_dist not in k_plus_nn
        ]
        k_plus_nn_candidate = nsmallest(1, point_idx_to_dist, key=lambda pair: pair[1])[
            0
        ]
    return [neighbour_idx for (neighbour_idx, _) in k_plus_nn]


def set_rknn_ti(
    points: List[Point],
    point_idx_ref_dist: List[Tuple[int, float]],
    k: int,
    m: float = 2.0,
    k_plus_nn_tolerance: float = 10e-9,
) -> None:
    for i in tqdm(range(len(points)), desc="Calculating rK+NN using TI..."):
        current_point_idx, current_point_ref_dist = point_idx_ref_dist[i]
        current_point = points[current_point_idx]

        k_plus_nn = get_knn_ti_indices(
            points=points,
            point_idx_ref_dist=point_idx_ref_dist,
            current_point=current_point,
            current_point_ref_dist_idx=i,
            k=k,
            m=m,
            k_plus_nn_tolerance=k_plus_nn_tolerance,
        )

        for idx in k_plus_nn:
            if points[idx].r_k_plus_nn is None:
                points[idx].r_k_plus_nn = [current_point]
            else:
                points[idx].r_k_plus_nn.append(current_point)

        current_point.k_plus_nn = [points[idx] for idx in k_plus_nn]

    for point in points:
        if point.r_k_plus_nn is None:
            point.r_k_plus_nn = []


def get_knn_ti_indices(
    points: list[Point],
    point_idx_ref_dist: list[tuple[int, float]],
    current_point: Point,
    current_point_ref_dist_idx: int,
    k: int,
    set_point_info: bool = True,
    m: float = 2.0,
    k_plus_nn_tolerance: float = 10e-9,
) -> list[int]:
    dist_fn = distance_fn_generator(m)
    _, current_point_ref_dist = point_idx_ref_dist[current_point_ref_dist_idx]

    prev_idx_diff = 1
    next_idx_diff = 1
    search_prev = (current_point_ref_dist_idx - prev_idx_diff) >= 0
    search_next = (current_point_ref_dist_idx + next_idx_diff) <= (len(points) - 1)

    candidate_point_real_dist: list[tuple[int, float]] = []
    eps = 0.0
    stop_search = False
    while len(candidate_point_real_dist) < k - 1 or not stop_search:
        if len(candidate_point_real_dist) == k - 1:
            eps = max(candidate_point_real_dist, key=lambda pair: pair[1])[1]
            if set_point_info:
                current_point.max_eps = eps

        if not search_prev and search_next:
            go_next = True
            pessimistic_estimation = (
                point_idx_ref_dist[current_point_ref_dist_idx + next_idx_diff][1]
                - current_point_ref_dist
            )

        elif search_prev and not search_next:
            go_next = False
            pessimistic_estimation = (
                current_point_ref_dist
                - point_idx_ref_dist[current_point_ref_dist_idx - prev_idx_diff][1]
            )

        elif search_prev and search_next:
            ref_dist_diff_next = (
                point_idx_ref_dist[current_point_ref_dist_idx + next_idx_diff][1]
                - current_point_ref_dist
            )
            ref_dist_diff_prev = (
                current_point_ref_dist
                - point_idx_ref_dist[current_point_ref_dist_idx - prev_idx_diff][1]
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
            current_ref_dist_idx = current_point_ref_dist_idx + next_idx_diff
        else:
            current_ref_dist_idx = current_point_ref_dist_idx - prev_idx_diff

        if len(candidate_point_real_dist) >= k - 1 and pessimistic_estimation > eps:
            stop_search = True

        else:
            current_candidate_idx = point_idx_ref_dist[current_ref_dist_idx][0]
            current_candidate_point = points[current_candidate_idx]
            current_point_real_dist = dist_fn(current_point, current_candidate_point)

            if len(candidate_point_real_dist) < k - 1:
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

                    eps = max(
                        nsmallest(
                            k,
                            [dist for _, dist in candidate_point_real_dist],
                        )
                    )
            if go_next:
                next_idx_diff += 1
                search_next = (current_point_ref_dist_idx + next_idx_diff) <= (
                    len(points) - 1
                )
            else:
                prev_idx_diff += 1
                search_prev = (current_point_ref_dist_idx - prev_idx_diff) >= 0

    # Determine final k+NN
    if len(candidate_point_real_dist) > k - 1:
        sorted_candidate_idx_to_real_dist = sorted(
            candidate_point_real_dist, key=lambda pair: pair[1]
        )
        k_plus_nn = sorted_candidate_idx_to_real_dist[: k - 1]
        for (
            candidate_idx,
            candidate_dist,
        ) in sorted_candidate_idx_to_real_dist[k - 1 :]:
            if abs(k_plus_nn[-1][1] - candidate_dist) <= k_plus_nn_tolerance:
                k_plus_nn.append((candidate_idx, candidate_dist))
            else:
                break
    else:
        k_plus_nn = candidate_point_real_dist

    if set_point_info:
        current_point.min_eps = eps

    return [neighbour_idx for (neighbour_idx, _) in k_plus_nn]
