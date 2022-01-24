import time
from typing import Callable, List

from tqdm import tqdm
from utils import Point, get_pairwise_distances


def dbscan(
    points: list[Point], min_samples: int, eps: float, m: float = 2
) -> dict[str, float]:
    start_time = time.perf_counter()
    pairwise_distances = get_pairwise_distances(points, m)
    point_distance_time = time.perf_counter() - start_time

    # Determine core points
    start_time = time.perf_counter()
    eps_neighbours_indices = [
        get_eps_neighbour_indices(i, pairwise_distances, eps)
        for i in range(len(points))
    ]
    eps_neighbourhood_assignment_time = time.perf_counter() - start_time

    # Group core points in clusters
    start_time = time.perf_counter()

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

    assign_clusters_dbscan(
        core_points=core_points,
        non_core_points=non_core_points,
        neighbours_getter=lambda p: p.eps_neigbours,
    )
    clustering_time = time.perf_counter() - start_time

    return {
        "2_sort_by_ref_point_distances": point_distance_time,
        "3_eps_neighborhood/rnn_calculation": eps_neighbourhood_assignment_time,
        "4_clustering": clustering_time,
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


def assign_clusters_dbscan(
    core_points: List[Point], non_core_points: List[Point], neighbours_getter: Callable
) -> None:
    current_cluster_id = 1
    for core_point in tqdm(core_points, desc="Assigning core points to clusters..."):
        if core_point.cluster_id == 0:
            core_neighbours = [
                p for p in neighbours_getter(core_point) if p.point_type == 1
            ]
            for p in core_neighbours:
                if p.cluster_id != 0:
                    core_point.cluster_id = p.cluster_id
                    break

            if core_point.cluster_id == 0:
                core_point.cluster_id = current_cluster_id
                current_cluster_id += 1

    # Assign cluster indices to non-core points
    for point in tqdm(non_core_points, desc="Assigning non-core points to clusters..."):
        for neighbour in neighbours_getter(point):
            if neighbour.point_type == 1:
                point.cluster_id = neighbour.cluster_id
                point.point_type = 0
                break
            if point.cluster_id == 0:
                point.point_type = -1
