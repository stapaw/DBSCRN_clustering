from dataclasses import dataclass, field
from typing import Callable, Optional, Union

import seaborn as sns
from tqdm import tqdm

sns.set_style("darkgrid")


@dataclass
class Example:
    id: Union[str, int]
    vals: list[float]
    ground_truth: int
    cluster_id: Optional[int] = None
    point_type: Optional[int] = None
    calc_ctr = 0
    debug_info: dict = field(default_factory=dict)

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
        values = "\t".join(
            [str(self.id)] + [str(val) for val in self.debug_info.values()]
        )
        return f"{values}\n"


def load_examples(dataset_path: str) -> list[Example]:
    gt_path = dataset_path.replace("points", "ground_truth")
    with open(dataset_path, "r") as f:
        points_lines = f.readlines()[1:]
    with open(gt_path, "r") as f:
        gt_lines = f.readlines()

    assert len(points_lines) == len(gt_lines)

    examples = []
    for id, (point_line, gt_line) in enumerate(zip(points_lines, gt_lines)):
        gt = int(gt_line.strip())
        point_coords = [float(val) for val in point_line.strip().split("\t")]
        examples.append(Example(id=id, vals=point_coords, ground_truth=gt))
    return examples


def distance_fn_generator(m: float) -> Callable[[Example, Example], float]:
    def distance(e1: Example, e2: Union[Example]) -> float:
        e1.calc_ctr += 1
        e2.calc_ctr += 1
        return sum(abs(e1.vals[i] - e2.vals[i]) ** m for i in range(len(e1.vals))) ** (
            1 / m
        )

    return distance


def get_pairwise_distances(
    examples: list[Example], m: float = 2
) -> dict[tuple[int, int], float]:
    dist_fn = distance_fn_generator(m)

    return {
        (i, j): dist_fn(examples[i], examples[j])
        for i in tqdm(range(len(examples)), desc="Calculating pairwise distances...")
        for j in range(i + 1, len(examples))
    }
