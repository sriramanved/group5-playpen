# Group 5: enclosure search experiment

`experiments-1` is a review branch for this approach. It is not the final deliverable.

`Player5(room, inventory, weights)` receives a room polygon,
counts of available wall/gate lengths and connector types, and scoring weights.
Its `build_enclosure()` method returns a `Construction` (starting point,
heading, ordered pieces, and one connector per vertex), or `None` if the
search finds no legal enclosure.

A legal enclosure closes inside the room without crossing itself, uses only
available stock, contains a gate, and has no straight face longer than 30 units.
A **face** is a run of collinear pieces joined by straight connectors. The score
is `1000 + A * area + C * perimeter`, plus `G` if at least two faces contain gates.
The perimeter weight `C` is nonpositive; the gate bonus `G` is nonnegative.

## Search procedure

A **seed** is an initial valid enclosure. The **frontier** is the list of valid
constructions waiting to be improved; it holds at most eight by default.

1. **Build seeds.** Try rectangles with one piece per face, then symmetric
   hexagons and octagons. Also combine collinear pieces into longer faces.
   Each face can contain at most six pieces because pieces are at least five
   units long. Packing searches keep independent inventory copies and remember
   failed combinations of remaining faces, stock, straight connectors, and gate
   usage. Each template keeps its first feasible packing.
2. **Find placements.** Try room-edge headings and a 15-degree heading sweep.
   Align shapes with room corners or place them at the interior centroid,
   representative point, and an 8-by-8 interior grid. Reject placements outside
   the buffered room's bounding box before checking the polygon. Keep up to two
   valid placements per packed shape. Once the frontier fills, skip seed templates
   whose score upper bound cannot reach its lowest score.
3. **Replace a boundary span.** Remove one or two adjacent pieces while keeping
   both endpoints and the remaining boundary fixed. Try paths of three or four
   pieces by default. Enumerate connector turns and the first piece lengths;
   solve for the final two lengths when their directions determine a unique
   solution. Reject unavailable lengths, insufficient stock, and paths whose
   score upper bound cannot beat the parent before checking geometry.
4. **Validate improvements.** Check face lengths, polygon validity, and room
   containment once per length path. Try available wall/gate assignments; the
   official validator checks each surviving construction. Add strictly
   higher-scoring children to the frontier and expand its highest score first.
   Deduplication includes piece boundaries, gate positions, and connector turns;
   only the choice of starting vertex is ignored. Retain the best valid result
   separately from the frontier.
5. **Use a spare gate.** When `G > 0`, replace a wall with an unused gate of the
   same length on a face without a gate. Keep the exchange only after validation
   confirms the bonus; it does not change the shape.

Containment allows the simulator's `1e-4` distance tolerance. Seed area limits
use the area of that same buffered room, so a legal near-boundary fit is kept.

The player measures CPU time with `time.process_time()`. Its default ceiling is
240 seconds; an active simulator timer can reduce this to 80% of the remaining
external allowance. By default, seed search gets at most six seconds or 35%
of the budget.
Each parent gets at most two seconds or 1/16 of a short budget, with a minimum
slice of 0.02 seconds; the overall deadline also applies. The player returns
its best result when the frontier empties or its budget ends. It never starts
another process.

This is a finite heuristic. It can miss placements, piece orderings, alternate
packings, non-improving intermediate moves, and paths longer than the selected
cap. Parents are not resumed after their time slice, and discarded frontier
states are not retried. Global optimality is not established.

## Reviewed-source verification

The [October 11 results](experiments/results_2026-10-11.csv) and
[run metadata](experiments/results_2026-10-11_metadata.json) record the reviewed
source after the clarity edits and containment-tolerance fix. It returned
35 valid enclosures and two `None` results across 37 cases, with no crashes or
timeouts. All 28 published group cases were valid. Total score was 215,543.94
in 605.27 aggregate CPU seconds; the longest case used 119.77 seconds.
All 34 tests and the official submission check passed.

The inputs and timing/scoring procedure are the same as described below.
This single run differs materially from the earlier measurements, so those
earlier percentage gains are not presented as this branch's measured margins.

## Archived performance measurements

The [37-case results](experiments/results_2026-10-09.csv) and
[measurement metadata](experiments/results_2026-10-09_metadata.json) record runs
from October 9, 2026, before the subsequent review edits. They describe the
recorded source hashes, not a new measurement of this branch. The earlier
uncommitted reference source and full reports are archived separately; exact
historical reruns require those sources.

Upstream commit `f94d6a90c0811ac719c7689fd304d6197fcfd8c1` supplies 28 published
group cases and seven shared cases. Our existing tiny case and extra rotated
room bring the comparison to 37. The official 30-case roster was not identified
in these files.

Each player/case pair ran once in a fresh process with a 300-second external
CPU allowance. Both search versions used a 240-second internal ceiling,
four-piece paths, and a frontier of eight. CPU includes player construction
and `build_enclosure()`; it excludes imports, scenario loading, and validation
after the player returns. Total score sums official case scores without
normalization. `None` and timeouts score zero; invalid returns score -1000.
Crashes remain labeled and count as zero only in comparison totals.

| Across 37 cases | Preserved baseline | Before optimization | After optimization |
|---|---:|---:|---:|
| Valid enclosures | 7 | 35 | 35 |
| Returned `None` | 4 | 2 | 2 |
| Timeouts | 23 | 0 | 0 |
| Crashes | 3 | 0 | 0 |
| Total score | 9,749.42 | 199,382.21 | 239,048.58 |
| Total CPU seconds | 7,349.15 | 714.21 | 355.39 |

Optimization checked stock before geometry, shared geometry across wall/gate
assignments, rejected impossible placements with bounding boxes, and cached
failed packing states. The observed total score rose 19.89% while total CPU
fell 50.24%: seven cases improved, 27 tied, and three regressed. The regressions
were shared `B` (-55), shared `invalid_examples` (-24.075), and Group 7
`horseshoe_room` (-20). The longest run increased from 90.15 to 140.86 CPU
seconds. These are single-run observations: time-based stopping can change the
candidates retained, so the margins are not fixed guarantees.

The same archived optimized source with a frontier of one scored 232,055.65 in
213.72 CPU seconds. Eight retained more alternatives and scored 3.01% higher
while using 66.29% more CPU; one is the faster option. Four-piece paths remain
the default. Optional five- and six-piece paths are available through
`local_path_cap`. They form a fixed outward rectangular bump replacing one
piece at least 15 units long; six pieces split the bump's middle segment.
Their larger search space can reduce progress within a time limit.

`baseline.py` preserves the original player from commit
`3decb8c126fe109995e4949cd5db838cd9754513`: an exhaustive wall-chain search with
one gate centered on the longest room edge and no internal deadline. The
original tests import this snapshot.

## Checks and reproduction

Mechanics tests cover stock backtracking, analytic path lengths, gate placement,
state identity, deadline behavior, failed-state reuse, and rotated-room fitting.
The extra [rotated room](../../scenarios/players/player5/rotated_diagonal_room.json)
is a regular octagon rotated 23 degrees, fitting within the simulator's
`1e-4` tolerance. A witness starts at `(12, 12)`, heading `23`, with one 5-unit
gate, seven 5-unit walls, and eight 45-degree turns. Its score is 1201.421356.

Run these commands from the repository root:

```sh
uv run python -m unittest players.player5.test_search players.player5.test_anytime
uv run python -m scripts.check_submission 5
uv run main.py --player 5 --scenario scenarios/players/player5/rotated_diagonal_room.json --seed 1

# Fetch the pinned input commit if it is not already available locally.
git fetch https://github.com/ryn-77/playpen.git f94d6a90c0811ac719c7689fd304d6197fcfd8c1

# Measure this branch on the pinned cases.
uv run python -m players.player5.benchmark --suite-ref f94d6a90c0811ac719c7689fd304d6197fcfd8c1 --algorithm incremental --cpu-limit 300 --workers 6 --output /tmp/group5-reviewed.json

# For future edits, save the current source before editing and compare afterward.
cp players/player5/player.py /tmp/group5-reference.py
uv run python -m players.player5.benchmark --suite-ref f94d6a90c0811ac719c7689fd304d6197fcfd8c1 --algorithm comparison --reference-player /tmp/group5-reference.py --cpu-limit 12 --workers 6 --output /tmp/group5-edit-comparison.json
```

The benchmark saves source hashes, outcomes, CPU time, search counts, and returned
constructions. Completed rows are checkpointed. Repeat `--case` to select exact
scenario paths, and use `--frontier-width` or `--path-cap` to compare settings.
Use `--algorithm both` to compare with the preserved original baseline.
Workers run independent cases, not a single player's search in parallel.
