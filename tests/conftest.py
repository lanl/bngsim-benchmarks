from __future__ import annotations

import copy
from pathlib import Path

import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]

VALID = {
    "id": "toy_model",
    "layer": "pilot",
    "title": "Toy model",
    "source": {
        "repo": "owner/models",
        "commit": "a" * 40,
        "path": "models/toy/toy.bngl",
        "sha256": "b" * 64,
        "format": "bngl",
    },
    "publications": [{"citation": "Author A (2000). A toy. J Toys 1:1.", "doi": "10.0/toy"}],
    "verification": {"kind": "pending", "evidence": "models/toy/verify.ipynb"},
    "capabilities": ["ode"],
    "protocol": {
        "ode": {
            "netgen": "generate_network({overwrite=>1})",
            "t_end": 600,
            "n_steps": 300,
            "rtol": 1.0e-8,
            "atol": 1.0e-8,
        }
    },
    "size": {"species": 9, "reactions": 14},
}


@pytest.fixture
def spec_data():
    """A fresh copy of a valid spec, to mutate one field at a time."""
    return copy.deepcopy(VALID)


@pytest.fixture
def write_spec(tmp_path):
    def _write(data, name=None, directory=None):
        d = Path(directory or tmp_path / "problems")
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name or data['id']}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write


VALID_TASK = {
    "id": "agent/timecourse/toy_peak",
    "category": "timecourse",
    "model": {"problem": "toy_model"},
    "forms": ["sbml", "bngl"],
    "request": "Simulate the model for 100 seconds. Report the peak of A and when it occurs.",
    "answer": [
        {"name": "peak_value", "reference": 0.0412, "rel_tol": 1.0e-3},
        {"name": "peak_time", "reference": 17.5, "abs_tol": 0.5},
    ],
    "expected_outcome": {"default": "answer"},
    "trap": "none",
    "split": "public",
}


@pytest.fixture
def task_data():
    """A fresh copy of a valid task, to mutate one field at a time."""
    return copy.deepcopy(VALID_TASK)


@pytest.fixture
def write_task(tmp_path):
    def _write(data, name=None, category=None, directory=None):
        parts = str(data.get("id", "agent/timecourse/toy_peak")).split("/")
        default = parts[1] if len(parts) == 3 else "timecourse"
        d = Path(directory or tmp_path / "tasks") / (category or default)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name or parts[-1]}.yaml"
        path.write_text(yaml.safe_dump(data, sort_keys=False))
        return path

    return _write
