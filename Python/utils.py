import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import seaborn as sns
from tqdm import tqdm

sns.set_style("darkgrid")


@dataclass
class Point:
    id: Union[str, int]
    vals: List[float]
    ground_truth: int = -1
    cluster_id: int = 0
    point_type: Optional[int] = None  # -1 for noise, 0 for non_core, 1 for core
    k_plus_nn: Optional[List["Point"]] = None
    r_k_plus_nn: Optional[List["Point"]] = None
    min_eps: Optional[float] = None
    max_eps: Optional[float] = None
    eps_neigbours: Optional[List["Point"]] = None
    calc_ctr: int = 0

    def __str__(self) -> str:
        return f"{self.id}({', '.join([str(round(val, 3)) for val in self.vals])})"

    def __repr__(self) -> str:
        return str(self)

    def serialize_out(self) -> str:
        return (
            f"{self.id},"
            f"{','.join([str(round(val, 3)) for val in self.vals])},"
            f"{self.calc_ctr},"
            f"{self.point_type},"
            f"{self.cluster_id}\n"
        )

    def serialize_debug(self) -> str:
        """
        :return: Returns info for creating debug file.
        """
        debug_info = [str(self.id)]
        if self.max_eps is not None:
            debug_info.append(str(round(self.max_eps, 3)))
        if self.min_eps is not None:
            debug_info.append(str(round(self.min_eps, 3)))
        if self.r_k_plus_nn is not None:
            r_k_plus_nn_ids = sorted(p.id for p in self.r_k_plus_nn)
            k_plus_nn_ids = sorted(p.id for p in self.k_plus_nn)
            debug_info.append(str(len(r_k_plus_nn_ids)))
            debug_info.append(str(k_plus_nn_ids))
            debug_info.append(str(r_k_plus_nn_ids))

        if self.eps_neigbours is not None:
            eps_neighbours_ids = sorted(p.id for p in self.eps_neigbours)
            debug_info.extend([str(eps_neighbours_ids), str(len(eps_neighbours_ids))])

        values = "\t".join(debug_info)
        return f"{values}\n"

    def get_serialize_debug_header(self) -> str:
        keys = ["id"]
        if self.r_k_plus_nn is not None:
            if self.min_eps is not None:
                keys.extend(["max_eps", "min_eps"])
            keys.extend(["|rk+NN|", "k+NN", "rk+NN"])
        else:
            keys.extend(["eps_neighbours", "|eps_neighbours|"])

        values = "\t".join(keys)
        return f"{values}\n"


def load_points(dataset_path: str) -> List[Point]:
    gt_path = dataset_path.replace("points", "ground_truth")
    with open(dataset_path, "r") as f:
        points_lines = f.readlines()[1:]
    with open(gt_path, "r") as f:
        gt_lines = f.readlines()

    assert len(points_lines) == len(gt_lines)

    points = []
    for idx, (point_line, gt_line) in enumerate(zip(points_lines, gt_lines)):
        gt = int(gt_line.strip())
        point_coords = [float(val) for val in point_line.strip().split("\t")]
        points.append(Point(id=idx, vals=point_coords, ground_truth=gt))
    return points


def distance_fn_generator(m: float) -> Callable[[Point, Point], float]:
    def distance(p1: Point, p2: Point) -> float:
        p1.calc_ctr += 1
        p2.calc_ctr += 1
        return sum(abs(p1.vals[i] - p2.vals[i]) ** m for i in range(len(p1.vals))) ** (
            1 / m
        )

    return distance


def get_pairwise_distances(
    points: List[Point],
    m: float = 2,
    verbose: bool = True,
    cache: Optional[Path] = None,
) -> Tuple[float, Dict[Tuple[int, int], float]]:
    start_time = time.perf_counter()
    if cache is not None and cache.exists():
        with cache.open("r") as f:
            json_distances = json.load(f)
            distances = {
                (int(str_keys.split(",")[0]), int(str_keys.split(",")[1])): dist
                for str_keys, dist in json_distances.items()
            }

    if cache is None or not cache.exists():
        dist_fn = distance_fn_generator(m)
        iterator = range(len(points))
        if verbose:
            iterator = tqdm(iterator, desc="Calculating pairwise distances...")
        distances = {
            (i, j): dist_fn(points[i], points[j])
            for i in iterator
            for j in range(i + 1, len(points))
        }

    point_distance_time = time.perf_counter() - start_time
    if cache is not None and not cache.exists():
        with cache.open("w+") as f:
            cache.parent.mkdir(parents=True, exist_ok=True)
            json_distances = {
                f"{indices_tuple[0]},{indices_tuple[1]}": round(dist, 3)
                for indices_tuple, dist in distances.items()
            }
            json.dump(json_distances, f, indent=2)

    return point_distance_time, distances
