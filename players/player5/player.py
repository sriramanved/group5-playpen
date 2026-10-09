"""Naive exhaustive search with a fixed first-gate pose and exactly one gate."""

import math
from dataclasses import dataclass

from shapely.geometry import LineString

from players.player0 import Player
from src.constants import ANGLE_TOL, MAX_FACE_LENGTH, MIN_ENCLOSURE_PIECES, TOL
from src.enclosure import Construction, score_construction, validate_construction
from src.pieces import Connector, ConnectorType, Piece, PieceType


@dataclass
class SearchStats:
    nodes: int = 0
    closed_candidates: int = 0
    valid_polygons: int = 0
    boundary_prunes: int = 0
    crossing_prunes: int = 0
    face_prunes: int = 0
    exhausted_leaves: int = 0


class Player5(Player):
    """Enumerate every wall/connector chain rooted at a fixed starting gate.

    Defaults to centering the gate along the longest room edge.
    Optional keyword arguments fix another pose or a particular gate length.
    No beam width, depth cap, early success exit, or internal time limit: the
    simulator's CPU limit still applies. Only tiny inventories are practical.
    """

    def __init__(
        self,
        room,
        inventory,
        weights,
        *,
        start=None,
        start_heading=None,
        gate_length=None,
    ):
        super().__init__(room, inventory, weights)
        corners = room.get_boundary_points()
        edge_start, edge_end = max(
            zip(corners, corners[1:] + corners[:1]),
            key=lambda edge: math.dist(*edge),
        )
        self.edge_midpoint = (
            (edge_start[0] + edge_end[0]) / 2,
            (edge_start[1] + edge_end[1]) / 2,
        )
        self.start = None if start is None else tuple(start)
        self.start_heading = (
            math.degrees(
                math.atan2(edge_end[1] - edge_start[1], edge_end[0] - edge_start[0])
            )
            if start_heading is None
            else start_heading
        )
        self.gate_length = gate_length
        self.stats = SearchStats()
        self.best_score = float("-inf")

    def build_enclosure(self) -> Construction | None:
        self.stats = SearchStats()
        self.best_score = float("-inf")
        best = None
        stock = self.inventory.copy()
        room_area = self.room.polygon.buffer(TOL)
        choices = [
            Connector(kind, reflex)
            for kind in ConnectorType
            for reflex in (
                (False,) if kind == ConnectorType.STRAIGHT else (False, True)
            )
        ]
        pieces = []
        # Internal joints only. The root's closing joint is chosen at closure.
        joints = []
        points = []
        start = None

        def endpoint(heading, length):
            theta = math.radians(heading)
            x, y = points[-1]
            return x + length * math.cos(theta), y + length * math.sin(theta)

        def search(heading, face_length):
            nonlocal best
            self.stats.nodes += 1
            closed = math.dist(points[-1], start) <= TOL
            if closed:
                if len(pieces) >= MIN_ENCLOSURE_PIECES:
                    for closing in choices:
                        if stock.connectors.get(closing.connector_type, 0) <= 0:
                            continue
                        gap = (
                            heading + closing.turn_angle() - self.start_heading
                        ) % 360
                        if min(gap, 360 - gap) > ANGLE_TOL:
                            continue
                        candidate = Construction(
                            start,
                            self.start_heading,
                            list(pieces),
                            [closing, *joints],
                        )
                        self.stats.closed_candidates += 1
                        result = validate_construction(
                            candidate, self.room, self.inventory
                        )
                        if result.valid:
                            self.stats.valid_polygons += 1
                            score = score_construction(result, self.weights)
                            if score > self.best_score:
                                best, self.best_score = candidate, score
                # Extending a closed loop would revisit a vertex: never simple.
                return

            # One extra connector is always required to close at the root.
            if sum(stock.walls.values()) == 0 or stock.count_connectors() < 2:
                self.stats.exhausted_leaves += 1
                return

            for joint in choices:
                if not stock.take_connector(joint.connector_type):
                    continue
                next_heading = heading + joint.turn_angle()
                for length, count in stock.walls.items():
                    if count <= 0:
                        continue
                    next_face = face_length + length if joint.is_straight() else length
                    if next_face > MAX_FACE_LENGTH + TOL:
                        self.stats.face_prunes += 1
                        continue
                    end = endpoint(next_heading, length)
                    if not room_area.covers(LineString([points[-1], end])):
                        self.stats.boundary_prunes += 1
                        continue
                    # Snap only the simplicity check to handle floating point
                    # closure. Keep actual coordinates for the official judge.
                    check_end = start if math.dist(end, start) <= TOL else end
                    if not LineString([*points, check_end]).is_simple:
                        self.stats.crossing_prunes += 1
                        continue
                    stock.walls[length] -= 1
                    pieces.append(Piece(PieceType.WALL, length))
                    joints.append(joint)
                    points.append(end)
                    search(next_heading, next_face)
                    points.pop()
                    joints.pop()
                    pieces.pop()
                    stock.walls[length] += 1
                stock.connectors[joint.connector_type] += 1

        for length, count in sorted(stock.gates.items()):
            if count <= 0 or (
                self.gate_length is not None and length != self.gate_length
            ):
                continue
            if length > MAX_FACE_LENGTH + TOL:
                self.stats.face_prunes += 1
                continue
            theta = math.radians(self.start_heading)
            start = (
                self.start
                if self.start is not None
                else (
                    self.edge_midpoint[0] - length * math.cos(theta) / 2,
                    self.edge_midpoint[1] - length * math.sin(theta) / 2,
                )
            )
            points[:] = [start]
            end = endpoint(self.start_heading, length)
            if not room_area.covers(LineString([start, end])):
                self.stats.boundary_prunes += 1
                continue
            stock.gates[length] -= 1
            pieces.append(Piece(PieceType.GATE, length))
            points.append(end)
            search(self.start_heading, length)
            points.pop()
            pieces.pop()
            stock.gates[length] += 1

        outcome = (
            f"best score {self.best_score:g}" if best is not None else "no solution"
        )
        self.note = (
            f"Exhaustive fixed-gate search: {self.stats.nodes} nodes, "
            f"{self.stats.valid_polygons} valid polygons; {outcome}"
        )
        return best
