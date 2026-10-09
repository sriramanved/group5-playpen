# Naive exhaustive fixed-gate search

Run the intentionally tiny example:

```sh
uv run main.py --player 5 --scenario scenarios/players/player5/tiny_exhaustive.json --seed 1
```

It has one 5-unit gate, three 5-unit walls, and four right connectors in a
10-by-10 room. The optimum is a 5-by-5 square: area 25, perimeter 20,
score `1000 + 2*25 - 20 = 1030`. The full search visits 5 nodes and finds
one valid polygon.

This simple version uses exactly one gate. By default it centers the gate
on the longest room edge and points along that edge. Tied edges use the first
one in the room's vertex order, including the closing edge. It tries every
available gate length, keeping each centered on the same edge midpoint. To fix a different pose or gate length:

```python
from players.player5.player import Player5

player = Player5(room, inventory, weights, start=(5, 5), start_heading=0, gate_length=5)
construction = player.build_enclosure()
print(player.stats, player.best_score, player.note)
```

For each gate root, depth-first search appends every available wall length
with every available connector, including both flipped orientations of right
and diagonal connectors. Identical inventory items are counted rather than
permuted as distinct objects. Every time a chain returns to its start, the
search tries each remaining closing connector, validates the entire polygon
with the simulator, scores it, and keeps the maximum. It evaluates closures
at every depth, so a solution may leave inventory unused. Ties keep the first
solution; no valid polygon returns `None`. Inventory is copied before use.

Obvious dead branches stop when they leave the room (checking the entire wall,
including in concave rooms), cross/touch earlier walls, exceed the 30-unit
straight-face limit, or lack walls/connectors. Touching the room boundary is
allowed. One connector is reserved for the closing joint. The official
validator also checks faces that wrap across the starting gate. A closed
chain cannot extend into a simple polygon, so it is a leaf.

The tree is traversed exhaustively without storing all nodes at once. There
is no beam, search budget, or early return on the first polygon. This is
exhaustive only for the chosen pose and one-gate model, not over continuous
positions/headings or constructions with extra gates. Runtime grows
exponentially; the simulator can time out on larger examples and then scores
0, discarding any solution found so far. The recursive traversal is intended
for small inventories. The default simulator CPU limit is 300 seconds; use
`--cpu-limit` when experimenting.

Tests include an independent unpruned enumeration of a tiny inventory to
check the maximum score, plus closure, resource, orientation, concave-room,
straight-face, and repeatability cases:

```sh
uv run python -m unittest players.player5.test_search
uv run python -m scripts.check_submission 5 --cpu-limit 1
```

The simulator imports Tkinter even in headless mode. Use a Python installation
with Tkinter available, as described in the repository setup instructions.
