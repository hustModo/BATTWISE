from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from app.backend.services import cluster_input_builder  # noqa: E402


def main() -> int:
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        statis = pd.DataFrame({
            "Cell_num": [2, 4, 5, 6],
            "OCV ": [3.3, 3.4, 3.5, 3.6],
            "IR": [8.0, 9.0, 10.0, 11.0],
            "Capacity": [2.0, 1.9, 1.8, 1.7],
        })
        ddia = pd.DataFrame({
            "Cell_num": [2, 4, 5, 6],
            "MMD_Loss": [0.1, 0.2, 0.3, 0.4],
        })
        statis_xlsx = root / "Statis data.xlsx"
        ddia_csv = root / "ddia.csv"
        out_xlsx = root / "cluster_input.xlsx"
        statis.to_excel(statis_xlsx, index=False)
        ddia.to_csv(ddia_csv, index=False)

        out = cluster_input_builder.build_cluster_input(
            statis_xlsx=statis_xlsx,
            ddia_csv=ddia_csv,
            out_xlsx=out_xlsx,
            cell_nums=[2, 4, 5, 6],
        )
        assert list(out.columns) == ["Cell_num", "OCV", "IR", "CDDI"], out.columns
        assert out["Cell_num"].tolist() == [2, 4, 5, 6]
        assert "Capacity" not in out.columns
        saved = pd.read_excel(out_xlsx)
        assert list(saved.columns) == ["Cell_num", "OCV", "IR", "CDDI"]
        print("PASSED")
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
