import time
from heapq import nsmallest
from typing import List, Optional

from tqdm import tqdm

from utils import Example, get_pairwise_distances


def get_rknn(
    examples: List[Example],
    pairwise_distances: dict[tuple[int, int], float],
    k: int,
) -> dict[int, list[int]]:
    r_knn: dict[int, list[int]] = {i: [] for i in range(len(examples))}
    for i in tqdm(range(len(examples)), desc="Calculating rKNN..."):
        distances = [
            (m if m != i else n, pairwise_distances[(m, n) if m < n else (n, m)])
            for m in range(len(examples))
            for n in range(m + 1, len(examples))
            if i in (m, n)
        ]

        knn = nsmallest(
            k - 1, iterable=distances, key=lambda pair: pair[1]
        )  # point is in it's rKNN

        examples[i].debug_info["knn"] = [
            examples[neighbour_idx].id for (neighbour_idx, _) in knn
        ]
        for (neighbour_idx, _) in knn:
            r_knn[neighbour_idx].append(i)

    for i, example in enumerate(examples):
        example.debug_info["rknn"] = [examples[idx].id for idx in r_knn[i]]
    return r_knn


def dbscanrn(
    examples: list[Example], k: int, m: float = 2
) -> tuple[list[int], dict[str, float]]:
    start_time = time.perf_counter()
    pairwise_distances = get_pairwise_distances(examples, m)
    pairwise_distances_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    rknns = get_rknn(examples, pairwise_distances, k)
    rknns_time = time.perf_counter() - start_time

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
        "Pairwise distances calculation runtime": pairwise_distances_time,
        "RkNN calculation runtime": rknns_time,
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
