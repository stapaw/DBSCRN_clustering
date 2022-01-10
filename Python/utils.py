from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional, Union

import seaborn as sns
from tqdm import tqdm

sns.set_style("darkgrid")


@dataclass
class Example:
    id: Union[str, int]
    vals: list[float]
    cluster_id: Optional[int] = None
    point_type: Optional[int] = None
    calc_ctr = 0
    debug_info: dict = field(default_factory=dict)
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
        values = "\t".join(
            [str(self.id)] + [str(val) for val in self.debug_info.values()]
        )
        return f"{values}\n"


def load_examples(dataset_path: str) -> list[Example]:
    if ".tsv" in dataset_path:
        return read_tsv_examples(dataset_path)
    elif ".arff" in dataset_path:
        return read_arff_examples(dataset_path)
    elif ".pa" in dataset_path or ".txt" in dataset_path:
        dataset_path = dataset_path.replace(".txt", "").replace(".pa", "")
        return read_pa_examples(dataset_path)
    else:
        raise ValueError("Unknown dataset")


def read_tsv_examples(examples_path: Union[str, Path]) -> list[Example]:
    input_path = Path(examples_path)
    examples = []
    with input_path.open("r") as f:
        for line in f:
            spt = line.strip().split("\t")
            name = spt[0]
            vals = [float(val) for val in spt[1:-1]]
            ground_truth = spt[-1]
            examples.append(Example(id=name, vals=vals, ground_truth=ground_truth))
    return examples


def read_arff_examples(examples_path: Union[str, Path]) -> list[Example]:
    input_path = Path(examples_path)
    with input_path.open("r") as f:
        lines = [line.strip().lower() for line in f.readlines() if line.strip()]

    start_idx = lines.index("@data") + 1
    examples = []
    for i, line in enumerate(lines[start_idx:]):
        spt = line.strip().split(",")
        name = str(i)
        vals = [float(val) for val in spt[:-1]]
        cls = spt[-1]
        examples.append(Example(id=name, vals=vals, ground_truth=cls))

    return examples


def read_pa_examples(examples_path: Union[str, Path]) -> list[Example]:
    input_path_txt = Path(str(examples_path) + ".txt")
    input_path_pa = Path(str(examples_path) + ".pa")

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
        Example(id=i, vals=line_vals, ground_truth=cls)
        for i, (line_vals, cls) in enumerate(zip(vals, classes))
    ]


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
