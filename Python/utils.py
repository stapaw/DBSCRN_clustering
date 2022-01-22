from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import seaborn as sns
from tqdm import tqdm

sns.set_style("darkgrid")


@dataclass
class Point:
    id: Union[str, int]
    vals: list[float]
    ground_truth: int
    cluster_id: int = 0
    visited: bool = False
    point_type: Optional[int] = None  # -1 for noise, 0 for non_core, 1 for core
    k_plus_nn: Optional[list["Point"]] = None
    r_k_plus_nn: Optional[list["Point"]] = None
    min_eps: Optional[float] = None
    max_eps: Optional[float] = None
    eps_neigbours: Optional[list["Point"]] = None
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
        If running DBSCRN, returns:
            id, k+NN, rk+NN, |rk+NN|, (min_eps, max_eps) - optionally if TI was used
        If running DBSCAN, returns:
            id, eps_neighbours, |eps_neighbours|
        """
        debug_info = [str(self.id)]
        if self.k_plus_nn is not None:
            k_plus_nn_ids = [p.id for p in self.r_k_plus_nn]
            debug_info.append(str(k_plus_nn_ids))
        if self.r_k_plus_nn is not None:
            r_k_plus_nn_ids = [p.id for p in self.r_k_plus_nn]
            debug_info.extend([str(r_k_plus_nn_ids), str(len(r_k_plus_nn_ids))])
        if self.min_eps is not None:
            debug_info.append(str(self.min_eps))
        if self.max_eps is not None:
            debug_info.append(str(self.max_eps))

        if self.eps_neigbours is not None:
            eps_neighbours_ids = [p.id for p in self.eps_neigbours]
            debug_info.extend([str(eps_neighbours_ids), str(len(eps_neighbours_ids))])

        values = "\t".join(debug_info)
        return f"{values}\n"

    def get_serialize_debug_header(self) -> str:
        keys = ["id"]
        if self.r_k_plus_nn is not None:
            keys.extend(["k+NN", "rk+NN", "|rk+NN|"])
            if self.min_eps is not None:
                keys.extend(["min_eps", "max_eps"])
        else:
            keys.extend(["eps_neighbours", "|eps_neighbours|"])

        values = "\t".join(keys)
        return f"{values}\n"


def load_points(dataset_path: str) -> list[Point]:
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
    points: list[Point], m: float = 2
) -> dict[tuple[int, int], float]:
    dist_fn = distance_fn_generator(m)

    return {
        (i, j): dist_fn(points[i], points[j])
        for i in tqdm(range(len(points)), desc="Calculating pairwise distances...")
        for j in range(i + 1, len(points))
    }
