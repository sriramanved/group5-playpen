"""Mechanics and fallback tests, independent of benchmark scores."""

import itertools
import math
import time
import unittest
from collections import Counter
from types import SimpleNamespace
from unittest.mock import patch

from shapely.geometry import Polygon
from shapely.prepared import prep

from players.player5.benchmark import BenchmarkJob, run_one
from players.player5.benchmark import main as run_benchmark
from players.player5.player import (
    Player5,
    SeedTemplate,
    headings,
    solve_last_two,
    splice,
)
from src.constants import TOL
from src.enclosure import (
    Construction,
    build_geometry,
    score_construction,
    validate_construction,
)
from src.inventory import Inventory
from src.limits import CpuLimit
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room
from src.scenario import read_scenario
from src.weights import Weights

RIGHT = ConnectorType.RIGHT
DIAGONAL = ConnectorType.DIAGONAL
STRAIGHT = ConnectorType.STRAIGHT


def wall(length):
    return Piece(PieceType.WALL, length)


def gate(length):
    return Piece(PieceType.GATE, length)


def rectangle(width=10, height=10, start=(5, 5)):
    return Construction(
        start,
        0,
        [gate(width), wall(height), wall(width), wall(height)],
        [Connector(RIGHT)] * 4,
    )


class AnytimeTests(unittest.TestCase):
    def make_player(self, walls=None, gates=None, connectors=None, **kwargs):
        return Player5(
            Room(Polygon([(0, 0), (30, 0), (30, 30), (0, 30)])),
            Inventory(
                walls if walls is not None else {5: 3},
                gates if gates is not None else {5: 1},
                connectors if connectors is not None else {RIGHT: 4},
            ),
            Weights(2, -1, 100),
            cpu_budget=0.25,
            **kwargs,
        )

    def check(self, player, construction):
        self.assertIsNotNone(construction, player.note)
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid, result.reason)
        return result

    def test_tiny_optimum_repeatability_and_inventory(self):
        player = self.make_player()
        original = player.inventory.to_dict()
        first = player.build_enclosure()
        self.assertEqual(
            score_construction(self.check(player, first), player.weights), 1030
        )
        self.assertEqual(player.build_enclosure(), first)
        self.assertEqual(player.inventory.to_dict(), original)

    def test_deadline_returns_valid_best_so_far(self):
        player = self.make_player(
            walls={5: 30, 10: 30, 15: 30},
            gates={5: 2},
            connectors={RIGHT: 30, STRAIGHT: 30, DIAGONAL: 30},
        )
        with CpuLimit(1):
            construction = player.build_enclosure()
        self.check(player, construction)
        self.assertLess(player.stats.cpu, 0.4)
        self.assertIsNotNone(player.stats.first_valid_cpu)

    def test_two_right_connectors_use_hexagon(self):
        player = self.make_player(
            walls={5: 4, 10: 1}, gates={10: 1}, connectors={RIGHT: 2, DIAGONAL: 4}
        )
        construction = player.build_enclosure()
        self.check(player, construction)
        self.assertEqual(len(construction.pieces), 6)

    def test_unusable_short_inventory_items_are_skipped(self):
        player = self.make_player(walls={1: 20, 4: 20, 5: 3})
        construction = player.build_enclosure()
        self.check(player, construction)
        self.assertTrue(all(p.length >= 5 for p in construction.pieces))

    def test_no_gate_and_no_corner_connectors(self):
        self.assertIsNone(self.make_player(gates={}).build_enclosure())
        self.assertIsNone(self.make_player(walls={}).build_enclosure())
        self.assertIsNone(self.make_player(connectors={STRAIGHT: 20}).build_enclosure())

    def test_signature_preserves_gate_locations_but_ignores_start_index(self):
        player = self.make_player(walls={10: 3}, gates={10: 1})
        original = rectangle()
        original_result = self.check(player, original)
        moved_gate = Construction(
            original.start,
            original.start_heading,
            [wall(10), wall(10), gate(10), wall(10)],
            list(original.connectors),
        )
        moved_result = self.check(player, moved_gate)
        self.assertTrue(original_result.polygon.equals(moved_result.polygon))
        self.assertNotEqual(
            player._signature(original, original_result),
            player._signature(moved_gate, moved_result),
        )
        for start_index in range(4):
            rotated = Construction(
                original_result.vertices[start_index],
                headings(original)[start_index],
                original.pieces[start_index:] + original.pieces[:start_index],
                original.connectors[start_index:] + original.connectors[:start_index],
            )
            self.assertEqual(
                player._signature(original, original_result),
                player._signature(rotated, self.check(player, rotated)),
            )

    def test_signature_preserves_collinear_piece_boundaries(self):
        player = self.make_player(
            walls={5: 1, 10: 4}, gates={5: 1}, connectors={RIGHT: 4, STRAIGHT: 2}
        )
        original = Construction(
            (5, 5),
            0,
            [gate(5), wall(10), wall(10), wall(5), wall(10), wall(10)],
            [
                Connector(kind)
                for kind in (RIGHT, STRAIGHT, RIGHT, RIGHT, STRAIGHT, RIGHT)
            ],
        )
        changed = Construction(
            original.start,
            original.start_heading,
            [gate(5), wall(10), wall(10), wall(10), wall(5), wall(10)],
            list(original.connectors),
        )
        original_result = self.check(player, original)
        changed_result = self.check(player, changed)
        self.assertLess(
            original_result.polygon.symmetric_difference(changed_result.polygon).area,
            TOL,
        )
        self.assertNotEqual(
            player._signature(original, original_result),
            player._signature(changed, changed_result),
        )

    def test_face_packing_recovers_from_a_stock_dead_end(self):
        player = self.make_player(
            walls={5: 1, 10: 2}, gates={5: 1}, connectors={RIGHT: 4, STRAIGHT: 1}
        )
        original_stock = player.inventory.to_dict()
        template = SeedTemplate(50, (10, 5, 10, 5), (2,) * 4)
        # Using the gate in a composite 10-unit face leaves no 5-unit pieces
        # for the short sides. Packing must backtrack to a single 10-unit wall.
        options = {10: [(gate(5), wall(5)), (wall(10),)], 5: [(gate(5),), (wall(5),)]}
        packed = player._pack_faces(template, options, time.process_time() + 0.1)
        self.check(player, packed)
        self.assertEqual(len(packed.pieces), 4)
        self.assertEqual(player.inventory.to_dict(), original_stock)

    def test_repeated_failed_packing_states_are_reused(self):
        player = self.make_player(
            walls={5: 4, 10: 1}, gates={5: 1}, connectors={RIGHT: 4, STRAIGHT: 8}
        )
        original_stock = player.inventory.to_dict()
        template = SeedTemplate(100, (10,) * 4, (2,) * 4)
        options = {10: [(wall(10),), (wall(5), wall(5)), (gate(5), wall(5))]}
        # Only 35 units of stock are available for a 40-unit boundary.
        # Swapping earlier face choices revisits the same impossible suffix.
        self.assertIsNone(
            player._pack_faces(template, options, time.process_time() + 1)
        )
        self.assertGreater(player.stats.packing_states_pruned, 0)
        self.assertEqual(player.inventory.to_dict(), original_stock)

    def test_invalid_benchmark_scenario_is_an_error_row(self):
        job = BenchmarkJob("malformed.json", {}, "incremental", 1, 0.5, 5, 8)
        row = run_one(job)
        self.assertEqual(row["status"], "error")
        self.assertIn("KeyError", row["detail"])
        self.assertEqual(row["score"], 0)
        self.assertEqual(row["cpu"], 0)

    def test_empty_benchmark_selection_is_rejected(self):
        args = SimpleNamespace(suite_ref=None, case=[], filter="missing")
        with (
            patch("players.player5.benchmark.parse_args", return_value=args),
            patch(
                "players.player5.benchmark.load_suite",
                return_value=(None, [("present.json", {})]),
            ),
            self.assertRaisesRegex(SystemExit, "no scenarios match"),
        ):
            run_benchmark()

    def test_nonfinite_budget_is_rejected(self):
        for budget in (float("inf"), float("nan")):
            with self.assertRaisesRegex(ValueError, "finite and positive"):
                Player5(
                    self.make_player().room,
                    Inventory(),
                    Weights(2, -1, 0),
                    cpu_budget=budget,
                )

    def test_analytic_solve_and_parallel_case(self):
        self.assertEqual(solve_last_two((15, 5), (1, 0), (0, 1)), (15, 5))
        self.assertIsNone(solve_last_two((15, 0), (1, 0), (-1, 0)))
        self.assertEqual(solve_last_two((10, 0), (0, -1), (1, 0)), (0, 10))

    def test_three_piece_extension_needs_both_endpoint_connectors(self):
        player = self.make_player(
            walls={10: 3, 5: 2}, gates={10: 1}, connectors={RIGHT: 4, STRAIGHT: 2}
        )
        original = rectangle(start=(0, 5))
        replacement = splice(
            original,
            0,
            1,
            [wall(5), gate(10), wall(5)],
            [Connector(STRAIGHT), Connector(RIGHT), Connector(RIGHT)],
            Connector(STRAIGHT),
        )
        result = self.check(player, replacement)
        self.assertEqual(len(replacement.connectors), len(replacement.pieces))
        self.assertAlmostEqual(result.area, 150)
        self.assertEqual(sum(c.is_straight() for c in replacement.connectors), 2)
        _, position_gap, heading_gap = build_geometry(replacement)
        self.assertLess(position_gap, TOL)
        self.assertEqual(heading_gap, 0)

    def test_five_piece_interior_bump_without_straight_connectors(self):
        player = self.make_player(
            walls={10: 2, 15: 1, 5: 5},
            gates={15: 1},
            connectors={RIGHT: 8},
            local_path_cap=5,
        )
        original = rectangle(15, 10, start=(0, 5))
        replacement = splice(
            original,
            2,
            1,
            [wall(5)] * 5,
            [
                Connector(RIGHT),
                Connector(RIGHT, True),
                Connector(RIGHT),
                Connector(RIGHT),
                Connector(RIGHT, True),
            ],
            Connector(RIGHT),
        )
        result = self.check(player, replacement)
        self.assertAlmostEqual(result.area, 175)
        player._prepared_room = prep(player.room.polygon.buffer(TOL))
        # Put the eligible 15-unit wall first so a deadline test does not
        # depend on how much earlier, unrelated edges cost to enumerate.
        rotated = Construction(
            (15, 15),
            180,
            [wall(15), wall(10), gate(15), wall(10)],
            [Connector(RIGHT)] * 4,
        )
        base = self.check(player, rotated)
        children = list(
            player.generate_local_replacements(
                rotated, base, time.process_time() + 0.25
            )
        )
        self.assertTrue(any(r.area > base.area for _, r in children))
        self.assertGreater(player.stats.valid_by_path.get(5, 0), 0)
        # With four new pieces, three internal +/-90 turns cannot sum to
        # zero. Five pieces are the minimum for this fixed-tangent bump.
        stock = Inventory({5: 5}, {}, {RIGHT: 6})
        self.assertFalse(
            any(
                dirs[0] == dirs[-1] == 0
                for _, _, dirs in player._patterns(4, 2, 2, stock)
            )
        )

    def test_second_gate_preserves_geometry_and_earns_exact_bonus(self):
        player = self.make_player(walls={10: 3}, gates={10: 2}, connectors={RIGHT: 4})
        original = rectangle()
        result = self.check(player, original)
        _changed, checked = player.try_second_gate(original, result)
        self.assertTrue(checked.second_gate_bonus)
        self.assertTrue(result.polygon.equals(checked.polygon))
        self.assertEqual(
            score_construction(checked, player.weights)
            - score_construction(result, player.weights),
            100,
        )
        player.weights = Weights(2, -1, 0)
        self.assertIs(player.try_second_gate(original, result)[0], original)

    def test_analytic_paths_match_independent_length_enumeration(self):
        player = self.make_player()
        counts = {wall(5): 3, gate(5): 1, wall(10): 2}
        start = (10, 10)
        cases = (
            ((-90, 0, 90), (5, 10, 5)),
            ((0, -90, 0, 90), (5, 5, 5, 5)),
            ((-45, 45, -45, 45), (5, 5, 5, 5)),
            ((0, -90, 0, 90, 0), (5, 5, 10, 5, 5)),
        )
        for angles, witness in cases:
            # Rotate away from the coordinate axes to exercise floating closure.
            directions = [
                (math.cos(math.radians(angle + 23)), math.sin(math.radians(angle + 23)))
                for angle in angles
            ]
            end = (
                start[0]
                + sum(
                    length * direction[0]
                    for length, direction in zip(witness, directions, strict=True)
                ),
                start[1]
                + sum(
                    length * direction[1]
                    for length, direction in zip(witness, directions, strict=True)
                ),
            )
            expected = set()
            for lengths in itertools.product((5, 10), repeat=len(directions)):
                if any(
                    needed > (4 if length == 5 else 2)
                    for length, needed in Counter(lengths).items()
                ):
                    continue
                endpoint = (
                    start[0]
                    + sum(
                        length * direction[0]
                        for length, direction in zip(lengths, directions, strict=True)
                    ),
                    start[1]
                    + sum(
                        length * direction[1]
                        for length, direction in zip(lengths, directions, strict=True)
                    ),
                )
                if math.dist(endpoint, end) < TOL:
                    expected.add(lengths)
            actual = {
                lengths
                for lengths, _ in player._replacement_paths(
                    start, end, directions, counts, time.process_time() + 1
                )
            }
            self.assertEqual(actual, expected)
            self.assertTrue(actual)

    def test_piece_assignments_preserve_stock_and_gate_requirement(self):
        player = self.make_player()
        counts = {wall(5): 2, gate(5): 1}
        assignments = list(
            player._piece_assignments((5, 5, 5), counts, needs_gate=True)
        )
        self.assertEqual(len(assignments), 3)
        for pieces in assignments:
            self.assertEqual(pieces.count(gate(5)), 1)
            self.assertEqual(pieces.count(wall(5)), 2)
        self.assertEqual(counts, {wall(5): 2, gate(5): 1})

    def test_seed_bound_skips_only_impossible_scores(self):
        template = SeedTemplate(25, (5,) * 4, (2,) * 4)
        options = {5: [(wall(5),), (gate(5),)]}
        player = self.make_player()
        player._seed_score_floor = 1040
        # One gate cannot earn the 100-point second-gate bonus. This seed's
        # upper score is 1030, so neither packing nor placement is necessary.
        with patch.object(player, "_pack_faces") as pack:
            self.assertEqual(
                list(
                    player._place_templates(
                        [template], options, {}, time.process_time() + 1, first_fit=True
                    )
                ),
                [],
            )
            pack.assert_not_called()
        self.assertEqual(player.stats.seed_bound_pruned, 1)
        # Two gates make 1130 attainable; a floor of 1080 must not prune it.
        player = self.make_player(gates={5: 2})
        player._prepare_placements()
        player._seed_score_floor = 1080
        results = list(
            player._place_templates(
                [template], options, {}, time.process_time() + 1, first_fit=True
            )
        )
        self.assertTrue(results)
        self.assertEqual(
            max(score_construction(result, player.weights) for _, result in results),
            1130,
        )

    def test_rotated_diagonal_room_has_a_witness_and_is_found(self):
        scenario = read_scenario("scenarios/players/player5/rotated_diagonal_room.json")
        player = Player5(
            scenario.room, scenario.inventory, scenario.weights, cpu_budget=0.25
        )
        witness = Construction(
            (12, 12), 23, [gate(5)] + [wall(5)] * 7, [Connector(DIAGONAL)] * 8
        )
        self.check(player, witness)
        self.check(player, player.build_enclosure())

    def test_seed_area_bound_respects_containment_tolerance(self):
        cases = (
            ("square", (RIGHT,) * 4),
            ("hexagon", (RIGHT, DIAGONAL, DIAGONAL) * 2),
            ("octagon", (DIAGONAL,) * 8),
        )
        for name, connector_types in cases:
            with self.subTest(shape=name):
                piece_count = len(connector_types)
                witness = Construction(
                    (0, 0),
                    0,
                    [gate(5)] + [wall(5)] * (piece_count - 1),
                    [Connector(kind) for kind in connector_types],
                )
                vertices, _, _ = build_geometry(witness)
                if name == "square":
                    room_polygon = Polygon(
                        [(0, 0), (4.99995, 0), (4.99995, 4.99995), (0, 4.99995)]
                    )
                else:
                    center = Polygon(vertices).centroid
                    # Shrink the exact enclosure about its center. The lost
                    # area exceeds TOL, but every vertex remains within TOL.
                    room_polygon = Polygon(
                        [
                            (
                                center.x + (x - center.x) * 0.99999,
                                center.y + (y - center.y) * 0.99999,
                            )
                            for x, y in vertices
                        ]
                    )
                player = Player5(
                    Room(room_polygon),
                    Inventory(
                        {5: piece_count - 1}, {5: 1}, dict(Counter(connector_types))
                    ),
                    Weights(2, -1, 100),
                    cpu_budget=0.25,
                )
                checked = self.check(player, witness)
                self.assertGreater(checked.area, room_polygon.area + TOL)
                player._prepare_placements()
                templates = player._seed_templates(
                    {5: [(wall(5),), (gate(5),)]}, time.process_time() + 1
                )
                self.assertTrue(
                    any(
                        template.face_lengths == (5,) * piece_count
                        for template in templates
                    )
                )
                self.check(player, player.build_enclosure())

    def test_impossible_headings_skip_polygon_checks(self):
        player = self.make_player()
        player.room = Room(Polygon([(0, 0), (20, 0), (20, 20), (0, 20)]))
        player._prepare_placements()
        # This rectangle has less area than the room, but even its best
        # rotation needs a bounding square wider than 20 units.
        shape = rectangle(30, 5, start=(0, 0))
        with patch.object(player._prepared_room, "covers") as covers:
            self.assertEqual(
                list(player.find_placements(shape, time.process_time() + 1)), []
            )
            covers.assert_not_called()
        self.assertGreater(player.stats.heading_bounds_pruned, 0)
        self.assertEqual(player.stats.validator_calls, 0)


if __name__ == "__main__":
    unittest.main()
