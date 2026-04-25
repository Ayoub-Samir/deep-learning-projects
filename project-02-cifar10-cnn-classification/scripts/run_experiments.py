from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.config import EXPERIMENT_SPECS, SEED
from src.experiment_runner import (
    prepare_experiment_context,
    run_experiment,
    write_experiment_reports,
)
from src.utils.common import set_global_seed


def main() -> None:
    set_global_seed(SEED)
    context = prepare_experiment_context()
    results: list[dict[str, object]] = []

    for spec in EXPERIMENT_SPECS:
        result = run_experiment(spec, context)
        results.append(result)
        write_experiment_reports(results)

    write_experiment_reports(results)


if __name__ == "__main__":
    main()
