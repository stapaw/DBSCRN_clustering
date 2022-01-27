from collections import defaultdict

from typing import Dict, List, Tuple

from scipy.special import comb
from tqdm import tqdm
from utils import Point, distance_fn_generator, get_pairwise_distances


def assert_gt_set(points: List[Point]) -> None:
    assert all(p.ground_truth is not None for p in points)


def assert_c_id_set(points: List[Point]) -> None:
    assert all(p.cluster_id != 0 for p in points)


def purity(points: List[Point]) -> float:
    """
    :param points: List of Points with cluster_id and ground_truth set.
    :return: Purity computed for the Points.
    """
    assert_gt_set(points)
    assert_c_id_set(points)

    gt_clusters = defaultdict(set)
    discovered_clusters = defaultdict(set)
    for p in tqdm(points, desc="Calculating purity..."):
        gt_clusters[p.ground_truth].add(p.id)
        discovered_clusters[p.cluster_id].add(p.id)

    total_card = 0
    for g_cid_set in gt_clusters.values():
        total_card += max(
            len(g_cid_set.intersection(c_cid_set))
            for c_cid_set in discovered_clusters.values()
        )
    return total_card / len(points)


def rand(points: List[Point]) -> Tuple[float, int, int, int]:
    """
    :param points: List of Points with cluster_id and ground_truth set.
    :return: Tuple with rand value, |tp|, |tn| and pairs count.
    """
    assert_gt_set(points)
    assert_c_id_set(points)

    count = comb(len(points), 2)
    tp = 0
    tn = 0
    for i in tqdm(range(len(points)), desc="Calculating RAND..."):
        for j in range(i + 1, len(points)):
            p1 = points[i]
            p2 = points[j]

            if p1.cluster_id == p2.cluster_id and p1.ground_truth == p2.ground_truth:
                tp += 1
            if p1.cluster_id != p2.cluster_id and p1.ground_truth != p2.ground_truth:
                tn += 1

    return (tp + tn) / count, tp, tn, count


def silhouette_coefficient(points: List[Point], m: float) -> float:
    assert_c_id_set(points)

    pairwise_distances = get_pairwise_distances(points, m)

    cluster_id_to_cluster_point_indices = defaultdict(list)
    for idx, point in enumerate(points):
        cluster_id_to_cluster_point_indices[point.cluster_id].append(idx)
    # Treat noise points as separate clusters
    try:
        noise_point_indices = cluster_id_to_cluster_point_indices.pop(-1)
        max_cluster_id = max(cluster_id_to_cluster_point_indices.keys())
        for idx in noise_point_indices:
            max_cluster_id += 1
            points[idx].cluster_id = max_cluster_id
            cluster_id_to_cluster_point_indices[max_cluster_id].append(idx)
    except KeyError:
        pass

    silhouette_coefficients = [None for _ in range(len(points))]
    for i, point in tqdm(
        enumerate(points),
        desc="Calculating silhouette coefficients...",
        total=len(points),
    ):
        same_cluster_point_indices = cluster_id_to_cluster_point_indices[
            point.cluster_id
        ]
        same_cluster_point_distances = [
            pairwise_distances[(i, j)]
            if (i, j) in pairwise_distances.keys()
            else pairwise_distances[(j, i)]
            for j in same_cluster_point_indices
            if j != i
        ]
        if len(same_cluster_point_distances) > 0:  # check for singleton clusters
            a = sum(same_cluster_point_distances) / len(same_cluster_point_distances)
        else:
            a = 0

        b_candidates = []
        for c_id, c_indices in cluster_id_to_cluster_point_indices.items():
            if c_id == point.cluster_id:
                continue
            other_cluster_point_indices = cluster_id_to_cluster_point_indices[c_id]
            other_cluster_point_distances = [
                pairwise_distances[(i, j)]
                if (i, j) in pairwise_distances.keys()
                else pairwise_distances[(j, i)]
                for j in other_cluster_point_indices
            ]
            b = sum(other_cluster_point_distances) / len(other_cluster_point_distances)
            b_candidates.append(b)
        b = min(b_candidates)

        silhouette_coefficients[i] = (b - a) / max(b, a)

    return sum(silhouette_coefficients) / len(silhouette_coefficients)


def davies_bouldin(points: List[Point], m: float) -> float:
    assert_c_id_set(points)

    dist_fn = distance_fn_generator(m)

    cluster_id_to_points = defaultdict(list)
    for point in points:
        cluster_id_to_points[point.cluster_id].append(point)

    cluster_centroids = []
    sigmas = []

    for cluster_id, cluster_points in cluster_id_to_points.items():
        centroid = Point(
            id=-1,
            vals=[
                sum(p.vals[dim] for p in cluster_points) / len(cluster_points)
                for dim in range(len(points[0].vals))
            ],
            cluster_id=cluster_id,
        )
        cluster_centroids.append(centroid)
        sigma = sum(dist_fn(centroid, p) for p in cluster_points) / len(cluster_points)
        sigmas.append(sigma)

    db_total = 0
    for i in range(len(cluster_centroids)):
        db_candidates = [
            (sigmas[i] + sigmas[j])
            / (dist_fn(cluster_centroids[i], cluster_centroids[j]))
            for j in range(len(cluster_centroids))
            if i != j
        ]
        db_total += max(db_candidates)

    return db_total / len(cluster_centroids)
