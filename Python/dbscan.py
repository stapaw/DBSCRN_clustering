import time
from typing import Optional

from tqdm import tqdm

from utils import Example, get_pairwise_distances


def dbscan(
    examples: list[Example], min_samples: int, eps: float, m: float = 2
) -> tuple[list[int], dict[str, float]]:
    start_time = time.perf_counter()
    pairwise_distances = get_pairwise_distances(examples, m)
    pairwise_distances_time = time.perf_counter() - start_time

    # Determine core points
    start_time = time.perf_counter()
    eps_neighbours_indices = [
        get_eps_neighbour_indices(i, pairwise_distances, eps)
        for i in tqdm(range(len(examples)), desc="Calculating core points...")
    ]
    core_point_indices = [
        i
        for i, i_eps_neighbours_indices in enumerate(eps_neighbours_indices)
        if len(i_eps_neighbours_indices)
        >= min_samples - 1  # in original algorithm, point belongs to it's own kNN
    ]
    core_points_assignment_time = time.perf_counter() - start_time

    for i in core_point_indices:
        examples[i].point_type = 1

    for i, i_eps_neighbours_indices in enumerate(eps_neighbours_indices):
        examples[i].debug_info["eps_neigbours"] = [
            examples[idx].id for idx in i_eps_neighbours_indices
        ]
        examples[i].debug_info["num_eps_neigh"] = len(i_eps_neighbours_indices)

    # Group core points in clusters
    start_time = time.perf_counter()
    cluster_ids: list[Optional[int]] = [None for _ in examples]
    current_cluster_id = 1
    for core_point_idx in tqdm(
        core_point_indices, desc="Assigning core points to clusters..."
    ):
        if cluster_ids[core_point_idx] is None:
            core_point_eps_neighbours_indices = eps_neighbours_indices[core_point_idx]
            eps_core_neighbours_indices = [
                index
                for index in core_point_eps_neighbours_indices
                if index in core_point_indices
            ]
            eps_core_neighbours_cluster_indices = [
                cluster_ids[i] for i in eps_core_neighbours_indices
            ]
            assigned_to_cluster = False
            for eps_neighbour_cluster_index in eps_core_neighbours_cluster_indices:
                if eps_neighbour_cluster_index is not None:
                    cluster_ids[core_point_idx] = eps_neighbour_cluster_index
                    assigned_to_cluster = True
            if not assigned_to_cluster:
                cluster_ids[core_point_idx] = current_cluster_id
                current_cluster_id += 1
    cluster_expansion_from_core_points_time = time.perf_counter() - start_time

    # Assign cluster indices to non-core points
    start_time = time.perf_counter()
    unassigned_points_indices = [
        idx for idx, cluster_id in enumerate(cluster_ids) if cluster_id is None
    ]
    for unassigned_point_index in tqdm(
        unassigned_points_indices, desc="Assigning non-core points to clusters..."
    ):
        unassigned_point_eps_neighbours_indices = eps_neighbours_indices[
            unassigned_point_index
        ]
        for neighbour_index in unassigned_point_eps_neighbours_indices:
            if neighbour_index in core_point_indices:
                cluster_ids[unassigned_point_index] = cluster_ids[neighbour_index]
                examples[unassigned_point_index].point_type = 0
                break
    border_points_cluster_assignment_time = time.perf_counter() - start_time

    start_time = time.perf_counter()
    noise_point_indices = [
        i for i in unassigned_points_indices if cluster_ids[i] is None
    ]
    final_ids = [c_id if c_id is not None else -1 for c_id in cluster_ids]
    noise_points_assignment_time = time.perf_counter() - start_time
    for idx in noise_point_indices:
        examples[idx].point_type = -1

    runtimes = {
        "Pairwise distance calculation runtime": pairwise_distances_time,
        "Core points assignment runtime": core_points_assignment_time,
        "Cluster expansion from core points runtime": cluster_expansion_from_core_points_time,
        "Border points assignment runtime": border_points_cluster_assignment_time,
        "Noise points assignment runtime": noise_points_assignment_time,
    }
    return final_ids, runtimes


def get_eps_neighbour_indices(
    root_idx: int, pairwise_distances: dict[tuple[int, int], float], eps: float
) -> list[int]:
    matching_pairs = [
        indices
        for indices, distance in pairwise_distances.items()
        if root_idx in indices and distance < eps
    ]
    return [
        indices[0] if indices[0] != root_idx else indices[1]
        for indices in matching_pairs
    ]
