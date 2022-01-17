from pathlib import Path

import pytest
from utils import load_examples

DATASETS_DIR = Path("datasets") / "points"


@pytest.mark.parametrize(
    "dataset_name",
    ["cluto-t7-10k.tsv", "complex9.tsv", "dim512.tsv", "example.tsv", "letter.tsv"],
)
def test_datasets(dataset_name: str):
    dataset_path = DATASETS_DIR / dataset_name

    assert dataset_path.exists()
    with dataset_path.open("r") as f:
        lines = f.readlines()

    header = lines[0]
    assert len(header.split("\t")) == 2
    lines = lines[1:]
    dims = set()
    for i, line in enumerate(lines):
        assert " " not in line, f"Line {i+1} contains whitespace separators"
        dims.add(len(line.split("\t")))
        assert (
            len(dims) <= 1
        ), f"Line {i + 1} is not consistent with the rest of the file"

    examples = load_examples(str(dataset_path))
    assert len(examples) > 0
