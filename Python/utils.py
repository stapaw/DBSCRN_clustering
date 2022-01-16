from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional, Union

import seaborn as sns
from tqdm import tqdm

sns.set_style("darkgrid")


@dataclass
class Point:
    id: Union[str, int]
    vals: list[float]
    cluster_id: int = 0
    visited: bool = False
    point_type: Optional[int] = None  # -1 for noise, 0 for non_core, 1 for core
    k_plus_nn: Optional[list["Point"]] = None
    r_k_plus_nn: Optional[list["Point"]] = None
    min_eps: Optional[float] = None
    max_eps: Optional[float] = None
    eps_neigbours: Optional[list["Point"]] = None
    calc_ctr: int = 0
    ground_truth: Optional[str] = None

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
    if ".tsv" in dataset_path:
        return read_tsv_points(dataset_path)
    elif ".arff" in dataset_path:
        return read_arff_points(dataset_path)
    elif ".pa" in dataset_path or ".txt" in dataset_path:
        dataset_path = dataset_path.replace(".txt", "").replace(".pa", "")
        return read_pa_points(dataset_path)
    else:
        raise ValueError("Unknown dataset")


def read_tsv_points(path: Union[str, Path]) -> list[Point]:
    input_path = Path(path)
    points = []
    with input_path.open("r") as f:
        for line in f:
            spt = line.strip().split("\t")
            name = spt[0]
            vals = [float(val) for val in spt[1:-1]]
            ground_truth = spt[-1]
            points.append(Point(id=name, vals=vals, ground_truth=ground_truth))
    return points


def read_arff_points(path: Union[str, Path]) -> list[Point]:
    input_path = Path(path)
    with input_path.open("r") as f:
        lines = [line.strip().lower() for line in f.readlines() if line.strip()]

    start_idx = lines.index("@data") + 1
    points = []
    for i, line in enumerate(lines[start_idx:]):
        spt = line.strip().split(",")
        name = str(i)
        vals = [float(val) for val in spt[:-1]]
        cls = spt[-1]
        points.append(Point(id=name, vals=vals, ground_truth=cls))

    return points


def read_pa_points(path: Union[str, Path]) -> list[Point]:
    input_path_txt = Path(str(path) + ".txt")
    input_path_pa = Path(str(path) + ".pa")

    vals = []
    with input_path_txt.open("r") as f:
        for line in f.readlines():
            if line.strip():
                line = line.strip()
                spt = line.split(" ")
                line_vals = [float(el) for el in spt if el.strip()]
                vals.append(line_vals)

    with input_path_pa.open("r") as f:
        lines = [line.strip() for line in f.readlines()]

    pa_start_idx = 0
    for i, line in enumerate(lines):
        if "-" in line:
            pa_start_idx = i + 1
            break

    classes = lines[pa_start_idx:]
    return [
        Point(id=i, vals=line_vals, ground_truth=cls)
        for i, (line_vals, cls) in enumerate(zip(vals, classes))
    ]


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
