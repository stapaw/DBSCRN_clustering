from collections import defaultdict
from math import comb

import numpy as np
from tqdm import tqdm

from utils import Example, distance_fn_generator, get_pairwise_distances


def assert_gt_set(examples: list[Example]) -> None:
    assert all(example.ground_truth is not None for example in examples)


def assert_c_id_set(examples: list[Example]) -> None:
    assert all(example.cluster_id is not None for example in examples)


def purity(examples: list[Example]) -> float:
    """
    :param examples: List of Examples with cluster_id and ground_truth set.
    :return: Purity computed for the examples.
    """
    assert_gt_set(examples)
    assert_c_id_set(examples)

    gt_clusters = defaultdict(set)
    discovered_clusters = defaultdict(set)
    for example in tqdm(examples, desc="Calculating purity..."):
        gt_clusters[example.ground_truth].add(example.id)
        discovered_clusters[example.cluster_id].add(example.id)

    return (
        1
        / len(examples)
        * sum(
            max(len(g.intersection(c)) for c in gt_clusters.values())
            for g in discovered_clusters.values()
        )
    )


def rand(examples: list[Example]) -> tuple[float, int, int, int]:
    """
    :param examples: List of Examples with cluster_id and ground_truth set.
    :return: Tuple with rand value, |tp|, |tn| and pairs count.
    """
    assert_gt_set(examples)
    assert_c_id_set(examples)

    count = comb(len(examples), 2)
    tp = 0
    tn = 0
    for i in tqdm(range(len(examples)), desc="Calculating RAND..."):
        for j in range(i + 1, len(examples)):
            e1 = examples[i]
            e2 = examples[j]

            if e1.cluster_id == e2.cluster_id and e1.ground_truth == e2.ground_truth:
                tp += 1
            if e1.cluster_id != e2.cluster_id and e1.ground_truth != e2.ground_truth:
                tn += 1

    return (tp + tn) / count, tp, tn, count


def pointwise_silhouette_coefficients(examples: list[Example], m: float) -> list[float]:
    assert_c_id_set(examples)

    cluster_id_to_cluster_points = defaultdict(set)

    for i, example in enumerate(examples):
        cluster_id_to_cluster_points[example.cluster_id].add(i)

    pairwise_distances = get_pairwise_distances(examples, m)
    silhouette_coeeficients = [0.0 for _ in range(len(examples))]
    for i, example in tqdm(
        enumerate(examples),
        desc="Calculating silhouette coefficients...",
        total=len(examples),
    ):
        current_cluster_id = example.cluster_id
        same_cluster_ids = cluster_id_to_cluster_points[current_cluster_id]
        distances = [
            (m if m != i else n, pairwise_distances[(m, n) if m < n else (n, m)])
            for m in range(len(examples))
            for n in range(m + 1, len(examples))
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
    examples: list[Example], m: float
) -> dict[int, float]:
    silhouette_coefficients = pointwise_silhouette_coefficients(examples, m)

    cluster_id_to_cluster_points = defaultdict(set)
    for i, example in enumerate(examples):
        cluster_id_to_cluster_points[example.cluster_id].add(i)

    return {
        cluster_id: sum(silhouette_coefficients[c] for c in coefficients)
        / len(coefficients)
        for cluster_id, coefficients in cluster_id_to_cluster_points.items()
        if cluster_id != -1
    }


def mean_silhouette_coefficient(examples: list[Example], m: float) -> float:
    silhouette_coefficients = pointwise_silhouette_coefficients(examples, m)

    return sum(silhouette_coefficients) / len(silhouette_coefficients)


def davies_bouldin(examples: list[Example], m: float) -> float:
    assert_c_id_set(examples)
    assert all(example.cluster_id != -1 for example in examples)

    dist_fn = distance_fn_generator(m)

    cluster_id_to_cluster_points = defaultdict(set)
    for i, example in enumerate(examples):
        cluster_id_to_cluster_points[example.cluster_id].add(i)

    centroids = {}
    sigmas = {}
    for i, cluster_point_indices in cluster_id_to_cluster_points.items():
        cluster_examples = [examples[j] for j in cluster_point_indices]
        centroid_vals = [
            sum(example.vals[dim] for example in cluster_examples)
            / len(cluster_examples)
            for dim in range(len(examples[0].vals))
        ]
        centroid = Example(id=-1, vals=centroid_vals)
        centroids[i] = centroid

        centroid_distances = [dist_fn(e, centroid) for e in cluster_examples]
        sigma = sum(centroid_distances) / len(centroid_distances)
        sigmas[i] = sigma

    db_candidates = [
        (sigmas[i] + sigmas[j]) / (dist_fn(centroids[i], centroids[j]))
        for i in cluster_id_to_cluster_points.keys()
        for j in cluster_id_to_cluster_points.keys()
        if i != j
    ]

    return max(db_candidates) / len(cluster_id_to_cluster_points)
