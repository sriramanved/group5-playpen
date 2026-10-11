# Parallel DFS with a fixed starting gate

Run the intentionally tiny example with a five-second CPU limit:

```sh
uv run main.py --player 5 --scenario scenarios/players/player5/tiny_exhaustive.json --cpu-limit 5 --seed 1
```

It has one 5-unit gate, three 5-unit walls, and four right connectors in a
10-by-10 room. The optimum is a 5-by-5 square: area 25, perimeter 20,
score `1000 + 2*25 - 20 = 1030`.

The default runs up to four worker processes. Each explores a disjoint subset
of the first connector/wall choices for each gate length. Normal DFS continues
below those first branches. Workers do not repeat each other's subtrees; each
initial gate root is visited by each worker, so aggregate node counts include
those repeated roots. Budgets are fixed per worker; an idle worker's unused
budget is not redistributed. Results are merged by maximum score.

The default CPU budget is **five seconds shared across workers**, not five
seconds per worker. The player reserves 0.5 seconds for orchestration, then
splits the remaining 4.5 seconds evenly. Each fresh worker counts Python
startup plus its search using `time.process_time()`. Budget checks occur at
nodes and branch attempts, so an individual operation can overshoot slightly.
The reported total CPU includes worker startup/search plus parent CPU;
worker teardown is outside that measurement. The simulator's printed CPU time
counts only the parent process, which is why the player separately reports
combined CPU in its note. Wall time also includes startup and scheduling.

When the internal budget expires, each worker returns its best valid polygon
so far. The parent returns the highest-scoring one. `search_complete` says
whether every partition was exhausted; otherwise the result is only the best
found within budget, not a proven optimum. If no valid polygon was found,
return `None`. The simulator's own `--cpu-limit` still applies separately to
the parent process. Match it to the player's budget when benchmarking.

This version uses exactly one gate. By default it centers the gate on the
longest room edge and points along that edge. Ties use the first edge in the
room's vertex order, including the closing edge. It tries every usable gate
length, keeping each centered on the same edge midpoint. To customize:

```python
from players.player5.player import Player5

player = Player5(
    room,
    inventory,
    weights,
    start=(5, 5),
    start_heading=0,
    gate_length=5,
    workers=4,
    cpu_limit=5,
)
construction = player.build_enclosure()
print(player.stats, player.best_score, player.search_complete, player.note)
```

For the original unbounded exhaustive serial search, pass `workers=1,
cpu_limit=None`. Programs using multiprocessing must call the player inside
an `if __name__ == "__main__":` guard. The simulator already does this.

At each node DFS appends every available wall length with every available
connector, including both flipped orientations of right and diagonal
connectors. Identical items are counted rather than permuted as distinct
objects. Every closed chain tries the remaining closing connectors, goes
through the simulator's full validator, and is scored. Closures are evaluated
at every depth, so solutions may leave inventory unused. Inventory is copied
before use. Lengths below the simulator's five-unit minimum are unusable and
skipped, including those present in some upstream scenarios.

Distance pruning rejects a branch when its endpoint's straight-line distance
back to the gate start exceeds the maximum total length of walls it could
still add (plus the simulator's closure tolerance). It sums the longest
remaining usable walls, limited to `remaining connectors - 1` wall slots,
reserving a connector for closure. For example, if the start is 18 units away
but the remaining walls can span at most 15 units, closure is impossible.
This is a safe upper bound by the triangle inequality; equality is allowed.
It ignores turns and obstacles, so it can retain impossible paths but cannot
exclude a valid completion. `stats.distance_prunes` counts rejected nodes.
Set `distance_pruning=False` or benchmark with `--no-distance-pruning` for an
otherwise identical comparison.

Dead branches stop when they leave the room (checking the whole wall even in
concave rooms), cross/touch earlier walls, exceed the 30-unit straight-face
limit, or lack walls/connectors. Touching the room boundary is allowed. One
connector is reserved for closure. The official validator checks faces that
wrap across the gate. A closed chain is a leaf.

Without a budget, enumeration is exhaustive for the chosen pose and one-gate
model, not over continuous placements or constructions with extra gates.
Runtime grows exponentially. Parallelism reduces elapsed time and visits
different subtrees sooner, but cannot guarantee a better score under the same
total CPU budget. Recursive traversal remains intended for small inventories.

Run tests and benchmark all player scenarios:

```sh
uv run python -m unittest players.player5.test_search
uv run python -m scripts.check_submission 5 --cpu-limit 5
uv run python -m players.player5.benchmark --workers 4 --cpu-limit 5
```

The benchmark writes scores, completion status, node counts, combined CPU,
simulator CPU, and wall time to `logs/player5-parallel-5s.json`. Pass scenario
paths to run a subset, `--workers 1` for serial comparison, or `--output PATH`
to change the report location.

The simulator imports Tkinter even in headless mode. Use a Python installation
with Tkinter available, as described in the repository setup instructions.
