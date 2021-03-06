from dataclasses import dataclass
from typing import Callable, List, Optional, Union

import seaborn as sns

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
    eps_neighbours: Optional[List["Point"]] = None
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

        if self.eps_neighbours is not None:
            eps_neighbours_ids = sorted([p.id for p in self.eps_neighbours])
            debug_info.extend([str(len(eps_neighbours_ids)), str(eps_neighbours_ids)])

        values = "\t".join(debug_info)
        return f"{values}\n"

    def get_serialize_debug_header(self) -> str:
        keys = ["id"]
        if self.r_k_plus_nn is not None:
            if self.min_eps is not None:
                keys.extend(["max_eps", "min_eps"])
            keys.extend(["|rk+NN|", "k+NN", "rk+NN"])
        else:
            keys.extend(["|eps_neighbours|", "eps_neighbours"])

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
        return sum(abs(p1.vals[i] - p2.vals[i]) ** m for i in range(len(p1.vals))) ** (
            1 / m
        )

    return distance
