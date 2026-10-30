import json
import math
from pathlib import Path
import sys


def check(directory):
    metrics = json.loads((directory / "metrics.json").read_text())
    failures = []
    for name in ("max_slew", "max_cap", "max_fanout", "setup_violation_count", "hold_violation_count"):
        count = metrics[f"finish__timing__drv__{name}"]
        if count != 0:
            failures.append(f"{name}={count}")
    for kind in ("setup", "hold"):
        slack = float(metrics[f"finish__timing__{kind}__ws"])
        if not math.isfinite(slack) or slack < 0:
            failures.append(f"{kind}_slack={slack}")
    if (directory / "router_drc.rpt").read_text().strip():
        failures.append("router DRC report is nonempty")
    if failures:
        raise ValueError("; ".join(failures))


if __name__ == "__main__":
    if sys.argv[1:] == ["--help"]:
        print("usage: python3 tools/check_layout.py <layout-directory>")
        sys.exit(0)
    if len(sys.argv) != 2:
        print("error: expected one layout directory")
        print("help: python3 tools/check_layout.py <layout-directory>")
        sys.exit(2)
    try:
        check(Path(sys.argv[1]))
    except (ValueError, KeyError, OSError) as error:
        print(f"error: layout checks failed: {error}")
        print("help: inspect metrics.json, timing.rpt and router_drc.rpt in the layout directory")
        sys.exit(1)
    print("layout_checks: setup, hold, slew, capacitance, fanout and router DRC passed")
