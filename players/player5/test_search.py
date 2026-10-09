"""Run with: uv run python -m unittest players.player5.test_search."""

import itertools
import unittest

from shapely.geometry import Polygon

from players.player5.player import Player5
from src.enclosure import Construction, score_construction, validate_construction
from src.inventory import Inventory
from src.pieces import Connector, ConnectorType, Piece, PieceType
from src.room import Room
from src.weights import Weights

RIGHT = ConnectorType.RIGHT
STRAIGHT = ConnectorType.STRAIGHT
DIAGONAL = ConnectorType.DIAGONAL


class ExhaustiveSearchTests(unittest.TestCase):
    def player(self, walls=None, gates=None, connectors=None, room=None, **kwargs):
        kwargs.setdefault("start", (0, 0))
        return Player5(
            Room(Polygon(room or [(0, 0), (20, 0), (20, 20), (0, 20)])),
            Inventory(
                walls if walls is not None else {5: 3},
                gates if gates is not None else {5: 1},
                connectors if connectors is not None else {RIGHT: 4},
            ),
            Weights(A=2, C=-1, G=100),
            **kwargs,
        )

    def validated(self, player):
        construction = player.build_enclosure()
        self.assertIsNotNone(construction, player.note)
        result = validate_construction(construction, player.room, player.inventory)
        self.assertTrue(result.valid, result.reason)
        return construction, result

    def test_default_gate_is_centered_on_longest_edge(self):
        # Longest edge closes the room ring; it is vertical and points south.
        room = [(0, 0), (8, 0), (8, 8), (0, 11)]
        player = self.player(room=room, start=None)
        construction, _ = self.validated(player)
        self.assertAlmostEqual(construction.start[0], 0)
        self.assertAlmostEqual(construction.start[1], 8)
        self.assertAlmostEqual(construction.start_heading, -90)

    def test_default_centers_each_gate_length_independently(self):
        for gate_length in (5, 10):
            with self.subTest(gate_length=gate_length):
                player = self.player(
                    walls={gate_length: 3}, gates={gate_length: 1}, start=None
                )
                construction, _ = self.validated(player)
                self.assertAlmostEqual(construction.start[0], (20 - gate_length) / 2)
                self.assertAlmostEqual(construction.start[1], 0)
                self.assertAlmostEqual(construction.start_heading, 0)

    def test_tiny_square_and_repeatability(self):
        player = self.player(room=[(0, 0), (5, 0), (5, 5), (0, 5)])
        original = player.inventory.to_dict()
        first, result = self.validated(player)
        self.assertAlmostEqual(result.area, 25)
        self.assertAlmostEqual(result.perimeter, 20)
        self.assertAlmostEqual(player.best_score, 1030)
        self.assertEqual(player.stats.valid_polygons, 1)
        self.assertEqual(player.inventory.to_dict(), original)
        self.assertEqual(player.build_enclosure(), first)
        self.assertEqual(player.inventory.to_dict(), original)

    def test_clockwise_orientation_and_custom_pose(self):
        player = self.player(start=(10, 10), start_heading=0)
        self.validated(player)
        self.assertEqual(player.stats.valid_polygons, 2)
        player = self.player(start=(0, 0), start_heading=90)
        construction, _ = self.validated(player)
        self.assertTrue(all(c.reflex for c in construction.connectors))

    def test_no_gate_no_walls_and_insufficient_connectors(self):
        for kwargs in ({"gates": {}}, {"walls": {}}, {"connectors": {RIGHT: 3}}):
            with self.subTest(kwargs=kwargs):
                self.assertIsNone(self.player(**kwargs).build_enclosure())

    def test_boundaries_and_outside_root(self):
        player = self.player(room=[(0, 0), (4, 0), (4, 4), (0, 4)])
        self.assertIsNone(player.build_enclosure())
        self.assertGreater(player.stats.boundary_prunes, 0)
        self.assertIsNone(self.player(start=(-1, 0)).build_enclosure())

    def test_nonconvex_room_rejects_segment_between_inside_endpoints(self):
        player = self.player(
            room=[(0, 0), (20, 0), (20, 5), (5, 5), (5, 20), (0, 20)],
            gates={20: 1},
            start=(0, 15),
            start_heading=-45,
        )
        self.assertIsNone(player.build_enclosure())
        self.assertEqual(player.stats.boundary_prunes, 1)

    def test_gate_length_selection_and_all_available_roots(self):
        player = self.player(gates={6: 1, 5: 1})
        construction, _ = self.validated(player)
        self.assertEqual(construction.pieces[0].length, 5)
        self.assertIsNone(
            self.player(gates={6: 1, 5: 1}, gate_length=6).build_enclosure()
        )
        self.assertIsNone(self.player(gate_length=7).build_enclosure())

    def test_diagonal_connectors(self):
        player = self.player(walls={5: 7}, connectors={DIAGONAL: 8}, start=(5, 5))
        construction, result = self.validated(player)
        self.assertEqual(len(construction.pieces), 8)
        self.assertAlmostEqual(result.area, 50 * (1 + 2**0.5))

    def test_reflex_corner_in_l_shaped_room(self):
        player = self.player(
            walls={5: 4, 10: 1},
            gates={10: 1},
            connectors={RIGHT: 6},
            room=[(0, 0), (10, 0), (10, 5), (5, 5), (5, 10), (0, 10)],
        )
        construction, result = self.validated(player)
        self.assertAlmostEqual(result.area, 75)
        self.assertTrue(any(c.reflex for c in construction.connectors))

    def test_overlong_root_and_wraparound_face(self):
        player = self.player(gates={31: 1})
        self.assertIsNone(player.build_enclosure())
        self.assertEqual(player.stats.face_prunes, 1)
        # Only possible rectangle has a 35-unit bottom face straddling the
        # starting gate. The official validator must reject that closing joint.
        player = self.player(
            walls={20: 1, 35: 1, 5: 2},
            gates={15: 1},
            connectors={STRAIGHT: 1, RIGHT: 4},
            room=[(0, 0), (40, 0), (40, 10), (0, 10)],
            start=(20, 0),
        )
        self.assertIsNone(player.build_enclosure())

    def test_maximum_matches_independent_unpruned_enumeration(self):
        player = self.player(walls={5: 4, 10: 1}, connectors={STRAIGHT: 2, RIGHT: 4})
        # Enumerate every multiset permutation and every joint combination at
        # every depth, without any of the search's geometry/resource pruning.
        wall_lengths = [5, 5, 5, 5, 10]
        options = [Connector(STRAIGHT), Connector(RIGHT), Connector(RIGHT, True)]
        scores = []
        for depth in range(2, len(wall_lengths) + 1):
            for lengths in set(itertools.permutations(wall_lengths, depth)):
                pieces = [Piece(PieceType.GATE, 5)] + [
                    Piece(PieceType.WALL, length) for length in lengths
                ]
                for joints in itertools.product(options, repeat=len(pieces)):
                    candidate = Construction((0, 0), 0, pieces, list(joints))
                    result = validate_construction(
                        candidate, player.room, player.inventory
                    )
                    if result.valid:
                        scores.append(score_construction(result, player.weights))
        self.assertGreater(len({round(score, 6) for score in scores}), 1)
        construction, _ = self.validated(player)
        self.assertAlmostEqual(player.best_score, max(scores))
        self.assertAlmostEqual(player.best_score, 1070)
        self.assertGreater(player.stats.valid_polygons, 1)
        self.assertTrue(any(c.is_straight() for c in construction.connectors))


if __name__ == "__main__":
    unittest.main()
