import time

from tqdm import tqdm
from utils import Point, get_pairwise_distances


def dbscan(
    points: list[Point], min_samples: int, eps: float, m: float = 2
) -> dict[str, float]:
    start_time = time.perf_counter()
    pairwise_distances = get_pairwise_distances(points, m)
    pairwise_distances_time = time.perf_counter() - start_time

    # Determine core points
    start_time = time.perf_counter()
    eps_neighbours_indices = [
        get_eps_neighbour_indices(i, pairwise_distances, eps)
        for i in range(len(points))
    ]

    core_point_indices = [
        i
        for i, i_eps_neighbours_indices in enumerate(eps_neighbours_indices)
        if len(i_eps_neighbours_indices)
        >= min_samples - 1  # in original algorithm, point belongs to it's own kNN
    ]

    core_points = [p for i, p in enumerate(points) if i in core_point_indices]
    non_core_points = [p for i, p in enumerate(points) if i not in core_point_indices]
    for core_point in core_points:
        core_point.point_type = 1
    for i, i_eps_neighbours_indices in enumerate(eps_neighbours_indices):
        points[i].eps_neigbours = [points[idx] for idx in i_eps_neighbours_indices]

    core_points_assignment_time = time.perf_counter() - start_time

    # Group core points in clusters
    start_time = time.perf_counter()
    current_cluster_id = 1
    for core_point in tqdm(core_points, desc="Assigning core points to clusters..."):
        if core_point.cluster_id == 0:
            core_neighbours = [p for p in core_point.eps_neigbours if p.point_type == 1]
            for p in core_neighbours:
                if p.cluster_id != 0:
                    core_point.cluster_id = p.cluster_id
                    break

            if core_point.cluster_id == 0:
                core_point.cluster_id = current_cluster_id
                current_cluster_id += 1

    cluster_expansion_from_core_points_time = time.perf_counter() - start_time

    # Assign cluster indices to non-core points
    start_time = time.perf_counter()
    for point in tqdm(non_core_points, desc="Assigning non-core points to clusters..."):
        for neighbour in point.eps_neigbours:
            if neighbour.point_type == 1:
                point.cluster_id = neighbour.cluster_id
                point.point_type = 0
                break
            if point.cluster_id == 0:
                point.point_type = -1
    non_core_points_processing_time = time.perf_counter() - start_time

    return {
        "Pairwise distance calculation runtime": pairwise_distances_time,
        "Core points assignment runtime": core_points_assignment_time,
        "Cluster expansion from core points runtime": cluster_expansion_from_core_points_time,
        "Border points assignment runtime": non_core_points_processing_time,
    }


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
