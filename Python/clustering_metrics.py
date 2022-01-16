from collections import defaultdict
from math import comb

from tqdm import tqdm

from utils import Point, distance_fn_generator, get_pairwise_distances


def assert_gt_set(points: list[Point]) -> None:
    assert all(p.ground_truth is not None for p in points)


def assert_c_id_set(points: list[Point]) -> None:
    assert all(p.cluster_id is not None for p in points)


def purity(points: list[Point]) -> float:
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

    return (
        1
        / len(points)
        * sum(
            max(len(g.intersection(c)) for c in gt_clusters.values())
            for g in discovered_clusters.values()
        )
    )


def rand(points: list[Point]) -> tuple[float, int, int, int]:
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


def pointwise_silhouette_coefficients(points: list[Point], m: float) -> list[float]:
    assert_c_id_set(points)

    cluster_id_to_cluster_points = defaultdict(set)

    for i, point in enumerate(points):
        cluster_id_to_cluster_points[point.cluster_id].add(i)

    pairwise_distances = get_pairwise_distances(points, m)
    silhouette_coeeficients = [0.0 for _ in range(len(points))]
    for i, point in tqdm(
        enumerate(points),
        desc="Calculating silhouette coefficients...",
        total=len(points),
    ):
        current_cluster_id = point.cluster_id
        same_cluster_ids = cluster_id_to_cluster_points[current_cluster_id]
        distances = [
            (m if m != i else n, pairwise_distances[(m, n) if m < n else (n, m)])
            for m in range(len(points))
            for n in range(m + 1, len(points))
            if i in (m, n)
        ]

        if current_cluster_id == -1:
            a = 0.0
        else:
            same_cluster_distances = [
                distance for j, distance in distances if j in same_cluster_ids
            ]
            a = sum(same_cluster_distances) / len(same_cluster_distances)

        b_candidates = []
        for cluster_id, cluster_points_indices in cluster_id_to_cluster_points.items():
            if cluster_id == -1:
                b_candidates.extend(
                    [
                        distance
                        for j, distance in distances
                        if j in cluster_points_indices and j != i
                    ]
                )
            else:
                if cluster_id == current_cluster_id:
                    continue
                else:
                    cluster_distances = [
                        distance
                        for j, distance in distances
                        if j in cluster_points_indices
                    ]
                    b_candidates.append(sum(cluster_distances) / len(cluster_distances))
        b = min(b_candidates)

        silhouette_coeeficients[i] = (b - a) / max(b, a)

    return silhouette_coeeficients


def cluster_wise_silhouette_coefficients(
    points: list[Point], m: float
) -> dict[int, float]:
    silhouette_coefficients = pointwise_silhouette_coefficients(points, m)

    cluster_id_to_cluster_points = defaultdict(set)
    for i, point in enumerate(points):
        cluster_id_to_cluster_points[point.cluster_id].add(i)

    return {
        cluster_id: sum(silhouette_coefficients[c] for c in coefficients)
        / len(coefficients)
        for cluster_id, coefficients in cluster_id_to_cluster_points.items()
        if cluster_id != -1
    }


def mean_silhouette_coefficient(points: list[Point], m: float) -> float:
    silhouette_coefficients = pointwise_silhouette_coefficients(points, m)

    return sum(silhouette_coefficients) / len(silhouette_coefficients)


def davies_bouldin(points: list[Point], m: float) -> float:
    assert_c_id_set(points)
    assert all(p.cluster_id != -1 for p in points)

    dist_fn = distance_fn_generator(m)

    cluster_id_to_cluster_points = defaultdict(set)
    for i, point in enumerate(points):
        cluster_id_to_cluster_points[point.cluster_id].add(i)

    centroids = {}
    sigmas = {}
    for i, cluster_point_indices in cluster_id_to_cluster_points.items():
        cluster_points = [points[j] for j in cluster_point_indices]
        centroid_vals = [
            sum(p.vals[dim] for p in cluster_points) / len(cluster_points)
            for dim in range(len(points[0].vals))
        ]
        centroid = Point(id=-1, vals=centroid_vals)
        centroids[i] = centroid

        centroid_distances = [dist_fn(p, centroid) for p in cluster_points]
        sigma = sum(centroid_distances) / len(centroid_distances)
        sigmas[i] = sigma

    db_candidates = [
        (sigmas[i] + sigmas[j]) / (dist_fn(centroids[i], centroids[j]))
        for i in cluster_id_to_cluster_points.keys()
        for j in cluster_id_to_cluster_points.keys()
        if i != j
    ]

    return max(db_candidates) / len(cluster_id_to_cluster_points)
