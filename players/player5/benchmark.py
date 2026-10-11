"""Compare players on identical scenarios in fresh, CPU-limited processes.

Example:
    uv run python -m players.player5.benchmark --suite-ref upstream/main \
        --algorithm both --output /tmp/group5-results.json

The suite ref is pinned to a commit before any scenario is read. Completed
results are checkpointed so long baseline runs do not lose earlier results.

Each row's cpu measures player construction and build_enclosure() inside the
CPU limit. Wall time also includes setup and independent output validation.
"""

import argparse
import hashlib
import importlib.util
import json
import math
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path

from src.enclosure import score_construction, validate_construction
from src.limits import CpuLimit, PlayerTimeout
from src.scenario import Scenario

ROOT = Path(__file__).resolve().parents[2]
BASELINE_COMMIT = "3decb8c126fe109995e4949cd5db838cd9754513"
EXTRA_SCENARIOS = {
    "scenarios/players/player5/tiny_exhaustive.json",
    "scenarios/players/player5/rotated_diagonal_room.json",
}


@dataclass(frozen=True)
class BenchmarkJob:
    scenario_name: str
    scenario_data: dict
    algorithm: str
    cpu_limit: float
    internal_budget: float
    path_cap: int
    frontier_width: int
    reference_player: str | None = None


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def load_suite(ref=None):
    if ref:
        commit = git("rev-parse", ref)
        paths = git("ls-tree", "-r", "--name-only", commit, "scenarios").splitlines()
        suite = [
            (path, json.loads(git("show", f"{commit}:{path}")))
            for path in paths
            if path.endswith(".json")
        ]
    else:
        commit = None
        suite = [
            (str(path.relative_to(ROOT)), json.loads(path.read_text()))
            for path in sorted((ROOT / "scenarios").rglob("*.json"))
        ]
    names = {name for name, _ in suite}
    for path in sorted((ROOT / "scenarios/players/player5").glob("*.json")):
        name = str(path.relative_to(ROOT))
        if name not in names:
            suite.append((name, json.loads(path.read_text())))
    return commit, sorted(suite)


def player_class(algorithm, reference_player=None):
    if algorithm == "reference":
        specification = importlib.util.spec_from_file_location(
            "player5_reference", reference_player
        )
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        return module.Player5
    if algorithm == "baseline":
        from players.player5.baseline import Player5

        return Player5
    if algorithm == "example":
        from players.example_player import ExamplePlayer

        return ExamplePlayer
    from players.player5.player import Player5

    return Player5


def run_one(job):
    """Return a result row even if scenario setup or the player raises."""
    cpu = CpuLimit(job.cpu_limit)
    player = construction = result = None
    scenario = None
    status, detail = "none", "no validated enclosure found"
    wall_start = time.monotonic()
    try:
        scenario = Scenario.from_dict(job.scenario_data)
        implementation = player_class(job.algorithm, job.reference_player)
        options = {}
        if job.algorithm in ("incremental", "seeds", "reference"):
            options = {
                "cpu_budget": job.internal_budget,
                "local_path_cap": job.path_cap,
                "frontier_width": job.frontier_width,
                "seeds_only": job.algorithm == "seeds",
            }
        with cpu:
            player = implementation(
                scenario.room.copy(),
                scenario.inventory.copy(),
                scenario.weights,
                **options,
            )
            construction = player.build_enclosure()
        if construction is not None:
            result = validate_construction(
                construction, scenario.room, scenario.inventory
            )
            status = "valid" if result.valid else "invalid"
            detail = result.reason
    except PlayerTimeout as error:
        status, detail = "timeout", str(error)
    except Exception as error:  # noqa: BLE001 -- record player crashes as result rows
        status, detail = "error", f"{type(error).__name__}: {error}"

    row = {
        "scenario": job.scenario_name,
        "algorithm": job.algorithm,
        "status": status,
        "score": score_construction(result, scenario.weights)
        if result is not None
        else 0.0,
        "cpu": cpu.used,
        "wall": time.monotonic() - wall_start,
        "detail": detail,
        "note": getattr(player, "note", None),
    }
    if player is not None and hasattr(player, "stats"):
        row["stats"] = asdict(player.stats)
    if status == "valid":
        row.update(
            area=result.area,
            perimeter=result.perimeter,
            second_gate_bonus=result.second_gate_bonus,
            construction={
                "start": construction.start,
                "start_heading": construction.start_heading,
                "pieces": [
                    {"kind": piece.piece_type.value, "length": piece.length}
                    for piece in construction.pieces
                ],
                "connectors": [
                    {"kind": connector.connector_type.value, "reflex": connector.reflex}
                    for connector in construction.connectors
                ],
            },
        )
    return row


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite-ref", help="read scenarios at this git ref without checking it out"
    )
    parser.add_argument(
        "--algorithm",
        choices=[
            "both",
            "comparison",
            "baseline",
            "incremental",
            "reference",
            "seeds",
            "example",
        ],
        default="both",
        help=(
            "both: original/current; comparison: saved/current; "
            "baseline: original; incremental: current; reference: saved; "
            "seeds: current without local search; example: example player"
        ),
    )
    parser.add_argument("--cpu-limit", type=float, default=300)
    parser.add_argument(
        "--budget",
        type=float,
        help="internal budget; defaults to 80%% of the CPU limit, at most 240s",
    )
    parser.add_argument("--path-cap", type=int, default=4)
    parser.add_argument("--frontier-width", type=int, default=8)
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="independent runs; never parallelizes a player",
    )
    parser.add_argument(
        "--filter", default="", help="literal substring of the scenario path"
    )
    parser.add_argument(
        "--case",
        action="append",
        default=[],
        help="exact scenario path; repeat to select a development subset",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--reference-player",
        type=Path,
        help="frozen Player5 source for an old/new comparison",
    )
    args = parser.parse_args()
    if not math.isfinite(args.cpu_limit) or args.cpu_limit <= 0:
        parser.error("--cpu-limit must be finite and positive")
    if args.budget is None:
        args.budget = min(240, 0.8 * args.cpu_limit)
    if not math.isfinite(args.budget) or not 0 < args.budget < args.cpu_limit:
        parser.error("--budget must be positive and below --cpu-limit")
    if not 3 <= args.path_cap <= 6:
        parser.error("--path-cap must be between 3 and 6")
    if args.frontier_width < 1 or args.workers < 1:
        parser.error("--frontier-width and --workers must be positive")
    if args.algorithm in ("comparison", "reference"):
        if args.reference_player is None or not args.reference_player.is_file():
            parser.error("--reference-player must name a saved player.py")
        args.reference_player = args.reference_player.resolve()
    return args


def source_hashes(algorithms, reference_player=None):
    paths = {
        "baseline": "players/player5/baseline.py",
        "example": "players/example_player.py",
        "incremental": "players/player5/player.py",
        "seeds": "players/player5/player.py",
        "benchmark": "players/player5/benchmark.py",
    }
    if reference_player is not None:
        paths["reference"] = reference_player
    return {
        algorithm: hashlib.sha256((ROOT / paths[algorithm]).read_bytes()).hexdigest()
        for algorithm in [*algorithms, "benchmark"]
    }


def main():
    args = parse_args()
    commit, suite = load_suite(args.suite_ref)
    if not set(args.case).issubset(name for name, _ in suite):
        raise SystemExit("--case contains an unknown scenario path")
    suite = [(name, data) for name, data in suite if args.filter in name]
    if args.case:
        suite = [(name, data) for name, data in suite if name in args.case]
    if not suite:
        raise SystemExit("no scenarios match the selected --case/--filter")
    if args.algorithm == "both":
        algorithms = ["baseline", "incremental"]
    elif args.algorithm == "comparison":
        algorithms = ["reference", "incremental"]
    else:
        algorithms = [args.algorithm]
    jobs = [
        BenchmarkJob(
            name,
            data,
            algorithm,
            args.cpu_limit,
            args.budget,
            args.path_cap,
            args.frontier_width,
            str(args.reference_player) if args.reference_player is not None else None,
        )
        for name, data in suite
        for algorithm in algorithms
    ]
    report = {
        "suite_commit": commit,
        "baseline_commit": BASELINE_COMMIT,
        "scenario_count": len(suite),
        "published_group_scenarios": sum(
            name.startswith("scenarios/players/") and name not in EXTRA_SCENARIOS
            for name, _ in suite
        ),
        "cpu_limit": args.cpu_limit,
        "internal_budget": args.budget,
        "path_cap": args.path_cap,
        "frontier_width": args.frontier_width,
        "source_sha256": source_hashes(algorithms, args.reference_player),
        "rows": [],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save(row):
        report["rows"].append(row)
        report["rows"].sort(key=lambda entry: (entry["scenario"], entry["algorithm"]))
        temporary = args.output.with_suffix(".tmp")
        temporary.write_text(json.dumps(report, indent=2) + "\n")
        temporary.replace(args.output)
        print(
            f"{row['algorithm']:11} {row['scenario']:68} "
            f"{row['status']:8} score={row['score']:10.3f} cpu={row['cpu']:.3f} {row['detail']}",
            flush=True,
        )

    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    # Fresh workers match separate simulator runs and avoid profiling signals
    # observed when timed-out baseline workers were reused for another case.
    with ProcessPoolExecutor(max_workers=args.workers, max_tasks_per_child=1) as pool:
        futures = [pool.submit(run_one, job) for job in jobs]
        for future in as_completed(futures):
            save(future.result())


if __name__ == "__main__":
    main()
