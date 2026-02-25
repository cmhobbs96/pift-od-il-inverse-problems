"""Phase A: run forward PIFT Poisson pipeline."""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from pipelines.phase_a import run_phase_a_forward_poisson


def main() -> None:
    result = run_phase_a_forward_poisson(output_root=ROOT / "outputs", save_outputs=True)
    print("Phase A forward run complete")
    print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()
