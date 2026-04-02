from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "raw"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORTS_DIR = OUTPUTS_DIR / "reports"

SEED = 42
VALIDATION_SIZE = 0.2

PENDIGITS_URLS = {
    "train": "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tra",
    "test": "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.tes",
    "names": "https://archive.ics.uci.edu/ml/machine-learning-databases/pendigits/pendigits.names",
}


@dataclass(frozen=True)
class ExperimentSpec:
    name: str
    family: str
    hidden_layers: tuple[int, ...]
    standardize: bool
    learning_rate: float
    batch_size: int
    max_epochs: int
    l2_lambda: float = 0.0

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


EXPERIMENT_SPECS = (
    ExperimentSpec(
        name="m1_numpy_raw_baseline",
        family="numpy",
        hidden_layers=(32,),
        standardize=False,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=120,
    ),
    ExperimentSpec(
        name="m2_numpy_standardized",
        family="numpy",
        hidden_layers=(32,),
        standardize=True,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=120,
    ),
    ExperimentSpec(
        name="m3_numpy_deeper",
        family="numpy",
        hidden_layers=(64, 32),
        standardize=True,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=160,
    ),
    ExperimentSpec(
        name="m4_numpy_regularized",
        family="numpy",
        hidden_layers=(64, 32),
        standardize=True,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=160,
        l2_lambda=1e-4,
    ),
    ExperimentSpec(
        name="m5_sklearn_replica",
        family="sklearn",
        hidden_layers=(64, 32),
        standardize=True,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=160,
        l2_lambda=1e-4,
    ),
    ExperimentSpec(
        name="m6_pytorch_replica",
        family="pytorch",
        hidden_layers=(64, 32),
        standardize=True,
        learning_rate=0.01,
        batch_size=64,
        max_epochs=160,
        l2_lambda=1e-4,
    ),
)
