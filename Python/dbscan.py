import time
from typing import Callable, Dict, List, Tuple

from tqdm import tqdm
from utils import Point


def dbscan(
    points: List[Point],
    pairwise_distances: Dict[Tuple[int, int], float],
    min_samples: int,
    eps: float,
) -> Dict[str, float]:
    # Account for eps neighbourhood of point containing point
    min_samples -= 1

    # Determine core points
    start_time = time.perf_counter()
    filtered_distances = {
        idx: dist for idx, dist in pairwise_distances.items() if dist < eps
    }
    eps_neighbours_indices = [
        get_eps_neighbour_indices(i, filtered_distances)
        for i in tqdm(range(len(points)), desc="Determining eps neighbourhoods...")
    ]
    eps_neighbourhood_assignment_time = time.perf_counter() - start_time

    # Group core points in clusters
    start_time = time.perf_counter()

    core_point_indices = [
        i
        for i, i_eps_neighbours_indices in enumerate(eps_neighbours_indices)
        if len(i_eps_neighbours_indices) >= min_samples
    ]
    core_points = [p for i, p in enumerate(points) if i in core_point_indices]
    non_core_points = [p for i, p in enumerate(points) if i not in core_point_indices]
    for i, i_eps_neighbours_indices in enumerate(eps_neighbours_indices):
        points[i].eps_neigbours = [points[idx] for idx in i_eps_neighbours_indices]

    assign_clusters_dbscan(
        core_points=core_points,
        non_core_points=non_core_points,
        neighbours_getter_cp=lambda p: p.eps_neigbours,
        neighbours_getter_ncp=lambda p: p.eps_neigbours,
    )
    clustering_time = time.perf_counter() - start_time

    return {
        "3_eps_neighborhood/rnn_calculation": eps_neighbourhood_assignment_time,
        "4_clustering": clustering_time,
    }


def get_eps_neighbour_indices(
    root_idx: int, pairwise_distances: Dict[Tuple[int, int], float]
) -> List[int]:
    matching_pairs = [
        indices
        for indices, distance in pairwise_distances.items()
        if root_idx in indices
    ]
    return [
        indices[0] if indices[0] != root_idx else indices[1]
        for indices in matching_pairs
    ]


def assign_clusters_dbscan(
    core_points: List[Point],
    non_core_points: List[Point],
    neighbours_getter_cp: Callable,
    neighbours_getter_ncp,
) -> None:
    for core_point in core_points:
        core_point.point_type = 1

    current_cluster_id = 1
    for core_point in tqdm(core_points, desc="Assigning core points to clusters..."):
        if core_point.cluster_id == 0:
            core_point.cluster_id = current_cluster_id
            queue = [
                p
                for p in neighbours_getter_cp(core_point)
                if (p.point_type == 1 and p.cluster_id == 0)
            ]
            while len(queue) > 0:
                new_points_to_expand = []
                for point_to_expand in queue:
                    point_to_expand.cluster_id = current_cluster_id
                    new_points_to_expand.extend(
                        [
                            p
                            for p in neighbours_getter_cp(point_to_expand)
                            if (
                                p.point_type == 1
                                and p.cluster_id == 0
                                and p not in new_points_to_expand
                            )
                        ]
                    )
                queue = new_points_to_expand

            current_cluster_id += 1

    # Assign cluster indices to non-core points
    for point in tqdm(non_core_points, desc="Assigning non-core points to clusters..."):
        for neighbour in neighbours_getter_ncp(point):
            if neighbour.point_type == 1:
                point.cluster_id = neighbour.cluster_id
                point.point_type = 0
                break
        if point.cluster_id == 0:
            point.point_type = -1
            point.cluster_id = -1
