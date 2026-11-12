import json
import math
from pathlib import Path
import sys


def check(directory, setup=True):
    """Raise on any failed physical check; setup=False treats setup slack as a measurement (fmax)."""
    metrics = json.loads((directory / "metrics.json").read_text())
    failures = []
    names = ("max_slew", "max_cap", "max_fanout", "hold_violation_count")
    for name in names + (("setup_violation_count",) if setup else ()):
        count = metrics[f"finish__timing__drv__{name}"]
        if count != 0:
            failures.append(f"{name}={count}")
    for kind in ("setup", "hold") if setup else ("hold",):
        slack = float(metrics[f"finish__timing__{kind}__ws"])
        if not math.isfinite(slack) or slack < 0:
            failures.append(f"{kind}_slack={slack}")
    if (directory / "router_drc.rpt").read_text().strip():
        failures.append("router DRC report is nonempty")
    if failures:
        raise ValueError("; ".join(failures))


if __name__ == "__main__":
    usage = "usage: python3 tools/check_layout.py [--fmax] <layout-directory>\n--fmax: setup slack is measured, not gated"
    if sys.argv[1:] == ["--help"]:
        print(usage)
        sys.exit(0)
    args = sys.argv[1:]
    fmax = args[:1] == ["--fmax"]
    args = args[1:] if fmax else args
    if len(args) != 1:
        print("error: expected one layout directory")
        print("help: " + usage.splitlines()[0])
        sys.exit(2)
    try:
        check(Path(args[0]), setup=not fmax)
    except (ValueError, KeyError, OSError) as error:
        print(f"error: layout checks failed: {error}")
        print(
            "help: inspect metrics.json, timing.rpt and router_drc.rpt in the layout directory"
        )
        sys.exit(1)
    print(
        "layout_checks: "
        + ("hold" if fmax else "setup, hold")
        + ", slew, capacitance, fanout and router DRC passed"
    )
