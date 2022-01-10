import sys
import time
from heapq import nsmallest
from typing import Callable, List, Optional

from tqdm import tqdm

from utils import Example, distance_fn_generator

sys.setrecursionlimit(10000)


def get_rknn_ti(
    examples: List[Example],
    example_idx_to_ref_distance: list[tuple[int, float]],
    k: int,
    dist_fn: Callable[[Example, Example], float],
    k_plus_nn_tolerance: float = 10e-9,
) -> dict[int, list[int]]:
    r_k_plus_nn: dict[int, list[int]] = {i: [] for i in range(len(examples))}
    for i in tqdm(range(len(examples)), desc="Calculating rKNN using TI..."):
        current_example_idx, current_ref_dist = example_idx_to_ref_distance[i]
        current_example = examples[current_example_idx]

        prev_idx_diff = 1
        next_idx_diff = 1
        search_prev = (i - prev_idx_diff) >= 0
        search_next = (i + next_idx_diff) <= (len(examples) - 1)

        candidate_example_idx_to_real_dist: list[tuple[int, float]] = []
        eps = 0.0
        stop_search = False
        while len(candidate_example_idx_to_real_dist) < k - 1 or not stop_search:
            if len(candidate_example_idx_to_real_dist) == k - 1:
                eps = max(candidate_example_idx_to_real_dist, key=lambda pair: pair[1])[
                    1
                ]
                current_example.debug_info["max_eps"] = eps

            if not search_prev and search_next:
                go_next = True
                pessimistic_estimation = (
                    example_idx_to_ref_distance[i + next_idx_diff][1] - current_ref_dist
                )

            elif search_prev and not search_next:
                go_next = False
                pessimistic_estimation = (
                    current_ref_dist - example_idx_to_ref_distance[i - prev_idx_diff][1]
                )

            elif search_prev and search_next:
                ref_dist_diff_next = (
                    example_idx_to_ref_distance[i + next_idx_diff][1] - current_ref_dist
                )
                ref_dist_diff_prev = (
                    current_ref_dist - example_idx_to_ref_distance[i - prev_idx_diff][1]
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
                len(candidate_example_idx_to_real_dist) >= k - 1
                and pessimistic_estimation > eps
            ):
                stop_search = True

            else:
                current_candidate_example_idx = example_idx_to_ref_distance[
                    current_ref_dist_idx
                ][0]
                current_candidate_example = examples[current_candidate_example_idx]
                current_example_real_dist = dist_fn(
                    current_example, current_candidate_example
                )

                if len(candidate_example_idx_to_real_dist) < k - 1:
                    candidate_example_idx_to_real_dist.append(
                        (current_candidate_example_idx, current_example_real_dist)
                    )
                else:
                    # Account for floating point errors
                    if (
                        current_example_real_dist < eps
                        or abs(current_example_real_dist - eps) <= k_plus_nn_tolerance
                    ):
                        candidate_example_idx_to_real_dist.append(
                            (current_candidate_example_idx, current_example_real_dist)
                        )

                        eps = max(
                            nsmallest(
                                k,
                                [
                                    dist
                                    for _, dist in candidate_example_idx_to_real_dist
                                ],
                            )
                        )
                if go_next:
                    next_idx_diff += 1
                    search_next = (i + next_idx_diff) <= (len(examples) - 1)
                else:
                    prev_idx_diff += 1
                    search_prev = (i - prev_idx_diff) >= 0

        # Determine k+NN
        if len(candidate_example_idx_to_real_dist) > k - 1:
            sorted_candidate_example_idx_to_real_dist = sorted(
                candidate_example_idx_to_real_dist, key=lambda pair: pair[1]
            )
            k_plus_nn = sorted_candidate_example_idx_to_real_dist[: k - 1]
            for (
                candidate_idx,
                candidate_dist,
            ) in sorted_candidate_example_idx_to_real_dist[k - 1 :]:
                if abs(k_plus_nn[-1][1] - candidate_dist) <= k_plus_nn_tolerance:
                    k_plus_nn.append((candidate_idx, candidate_dist))
                else:
                    break
        else:
            k_plus_nn = candidate_example_idx_to_real_dist

        k_plus_nn_indices = [idx for idx, _ in k_plus_nn]

        for neighbour_idx in k_plus_nn_indices:
            r_k_plus_nn[neighbour_idx].append(current_example_idx)

        current_example.debug_info["min_eps"] = eps
        current_example.debug_info["K+nn"] = [
            examples[neighbour_idx].id for neighbour_idx in k_plus_nn_indices
        ]

    for i, example in enumerate(examples):
        example.debug_info["Rk+NN"] = [examples[idx].id for idx in r_k_plus_nn[i]]
        example.debug_info["|Rk+NN|"] = len(r_k_plus_nn[i])
    return r_k_plus_nn


def dbscanrn_ti(
    examples: list[Example], k: int, m: float = 2
) -> tuple[list[int], dict[str, float]]:
    """
    :param examples: Input examples.
    :param k: Number of the nearest neighbours. It is assumed that point is in it's k neighbours.
    :param m: Power used in Minkowsky distance function.
    """
    start_time = time.perf_counter()
    dist_fn = distance_fn_generator(m)
    ref_point = Example(id=-1, vals=[0.0 for _ in examples[0].vals])
    ref_distances = sorted(
        [
            (i, dist_fn(ref_point, e))
            for i, e in tqdm(
                enumerate(examples), desc="Calculating reference distances..."
            )
        ],
        key=lambda pair: pair[1],
    )
    reference_distances_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    rknns = get_rknn_ti(examples, ref_distances, k, dist_fn)
    rknns_ti_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    s_core = [i for i, rknn in rknns.items() if len(rknn) >= k]
    s_non_core = [i for i, rknn in rknns.items() if len(rknn) < k]
    core_points_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    current_cluster_idx = 1
    cluster_ids: list[Optional[int]] = [None for _ in range(len(examples))]
    for idx in tqdm(s_core, desc="Assigning core points to clusters..."):
        if cluster_ids[idx] is None:
            expand_cluster(idx, rknns, s_core, cluster_ids, current_cluster_idx)
            current_cluster_idx += 1
    final_ids = [c_id if c_id is not None else -1 for c_id in cluster_ids]
    cluster_assignment_time = time.perf_counter() - start_time

    for i, example in enumerate(examples):
        if i in s_core:
            example.point_type = 1
        else:
            example.point_type = 0

    runtimes = {
        "Reference point distances calculation runtime": reference_distances_time,
        "RkNN calculation with TI runtime": rknns_ti_time,
        "Core point calculation runtime": core_points_time,
        "Cluster assignment runtime": cluster_assignment_time,
    }
    return final_ids, runtimes


def expand_cluster(
    i: int,
    rknns: dict[int, list[int]],
    s_core: list[int],
    cluster_ids: list[Optional[int]],
    new_cluster_idx: int,
) -> None:
    cluster_ids[i] = new_cluster_idx
    unassigned_rknn_i = [idx for idx in rknns[i] if cluster_ids[idx] is None]

    for rknn_idx in unassigned_rknn_i:
        if rknn_idx in s_core:
            expand_cluster(rknn_idx, rknns, s_core, cluster_ids, new_cluster_idx)
        else:
            cluster_ids[rknn_idx] = new_cluster_idx
