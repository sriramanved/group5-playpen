"""Find a valid enclosure first, then improve short stretches of its boundary.

The original fixed-position exhaustive player is preserved in baseline.py.
Angles in turn patterns are multiples of 45 degrees. Construction headings
use degrees, as required by the simulator.
"""

import itertools
import math
import signal
import time
from collections import Counter
from dataclasses import dataclass, field

from shapely.geometry import Point, Polygon
from shapely.prepared import prep

from players.player0 import Player
from src.constants import BASELINE_SCORE, MAX_FACE_LENGTH, MIN_WALL_LENGTH, TOL
from src.enclosure import (
    Construction,
    ValidationResult,
    build_geometry,
    compute_faces,
    score_construction,
    validate_construction,
)
from src.pieces import Connector, ConnectorType, Piece, PieceType

RIGHT = ConnectorType.RIGHT
DIAGONAL = ConnectorType.DIAGONAL
STRAIGHT = ConnectorType.STRAIGHT
# Integer keys count 45-degree turns; negative turns use reflex connectors.
TURN = {
    0: Connector(STRAIGHT),
    1: Connector(DIAGONAL),
    -1: Connector(DIAGONAL, True),
    2: Connector(RIGHT),
    -2: Connector(RIGHT, True),
}


@dataclass
class SearchStats:
    local_candidates: int = 0
    inventory_pruned: int = 0
    geometry_pruned: int = 0
    validator_calls: int = 0
    valid_children: int = 0
    unique_states_retained: int = 0  # Cumulative admissions, not current frontier size.
    seeds_validated: int = 0
    seed_attempts: int = 0
    seed_bound_pruned: int = 0
    heading_bounds_pruned: int = 0
    placement_bounds_pruned: int = 0
    packing_states_pruned: int = 0
    second_gate_upgrades: int = 0
    expansions: int = 0
    patterns: int = 0
    candidates_by_path: dict = field(default_factory=dict)
    valid_by_path: dict = field(default_factory=dict)
    first_valid_cpu: float | None = None
    seed_cpu: float = 0.0
    cpu: float = 0.0
    best_by_depth: dict = field(default_factory=dict)


@dataclass(frozen=True)
class SeedTemplate:
    area: float
    face_lengths: tuple[int, ...]
    corner_turns: tuple[int, ...]


@dataclass
class SearchState:
    construction: Construction
    result: ValidationResult
    score: float
    depth: int


def headings(construction):
    """Return the outgoing heading of each piece, in degrees."""
    angles = [construction.start_heading]
    for connector in construction.connectors[1:]:
        angles.append(angles[-1] + connector.turn_angle())
    return angles


def solve_last_two(displacement, first_direction, second_direction):
    """Solve length1 * direction1 + length2 * direction2 = displacement."""
    first_x, first_y = first_direction
    second_x, second_y = second_direction
    determinant = first_x * second_y - first_y * second_x
    if abs(determinant) < 1e-10:
        return None  # Parallel directions do not determine two unique lengths.
    delta_x, delta_y = displacement
    return (
        (delta_x * second_y - delta_y * second_x) / determinant,
        (first_x * delta_y - first_y * delta_x) / determinant,
    )


def cross_sum(points):
    """Shoelace contribution of an open boundary path."""
    return sum(
        start_x * end_y - start_y * end_x
        for (start_x, start_y), (end_x, end_y) in itertools.pairwise(points)
    )


def available_pieces(stock):
    """Legal pieces and their counts, with longer walls considered first."""
    return {
        Piece(kind, length): count
        for kind, pool in (
            (PieceType.WALL, stock.walls),
            (PieceType.GATE, stock.gates),
        )
        for length, count in sorted(pool.items(), reverse=True)
        if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
    }


def splice(construction, edge, removed, pieces, new_connectors, terminal_connector):
    """Replace adjacent pieces, rotating the start so the span never wraps.

    The connector at endpoint B belongs to the first unchanged piece.
    Therefore a path of n new pieces needs n+1 connectors, at A through B.
    """
    vertices, _, _ = build_geometry(construction)
    piece_headings = headings(construction)
    old_pieces = construction.pieces[edge:] + construction.pieces[:edge]
    old_connectors = construction.connectors[edge:] + construction.connectors[:edge]
    incoming_heading = piece_headings[edge] - old_connectors[0].turn_angle()
    return Construction(
        vertices[edge],
        incoming_heading + new_connectors[0].turn_angle(),
        list(pieces) + old_pieces[removed:],
        list(new_connectors) + [terminal_connector] + old_connectors[removed + 1 :],
    )


class Player5(Player):
    def __init__(
        self,
        room,
        inventory,
        weights,
        *,
        cpu_budget=240.0,
        local_path_cap=4,
        frontier_width=8,
        seeds_only=False,
    ):
        super().__init__(room, inventory, weights)
        if not math.isfinite(cpu_budget) or cpu_budget <= 0:
            raise ValueError("cpu_budget must be finite and positive")
        if not isinstance(local_path_cap, int) or not 3 <= local_path_cap <= 6:
            raise ValueError("local_path_cap must be an integer from 3 through 6")
        if not isinstance(frontier_width, int) or frontier_width < 1:
            raise ValueError("frontier_width must be a positive integer")
        self.cpu_budget = float(cpu_budget)
        self.local_path_cap = local_path_cap
        self.frontier_width = frontier_width
        self.seeds_only = seeds_only
        self.stats = SearchStats()
        self.best_score = float("-inf")
        self._pattern_cache = {}
        self._seed_score_floor = float("-inf")
        legal_gates = sum(
            count
            for length, count in inventory.gates.items()
            if count > 0 and MIN_WALL_LENGTH <= length <= MAX_FACE_LENGTH
        )
        self._possible_gate_bonus = weights.G if legal_gates >= 2 else 0

    def _validate(self, construction):
        self.stats.validator_calls += 1
        result = validate_construction(construction, self.room, self.inventory)
        return result if result.valid else None

    def _remaining(self, construction):
        """Return unused stock. Search parents have already passed validation."""
        stock = self.inventory.copy()
        for piece in construction.pieces:
            if not stock.take_piece(piece):
                raise ValueError("construction exceeds the piece inventory")
        for connector in construction.connectors:
            if not stock.take_connector(connector.connector_type):
                raise ValueError("construction exceeds the connector inventory")
        return stock

    def try_second_gate(self, construction, result):
        """Earn the bonus by exchanging a wall for a spare same-length gate."""
        if self.weights.G == 0 or result.second_gate_bonus:
            return construction, result
        stock = self._remaining(construction)
        for face in result.faces:
            if face.has_gate:
                continue
            for piece_index in face.piece_indices:
                piece = construction.pieces[piece_index]
                if stock.gates.get(piece.length, 0) <= 0:
                    continue
                changed_pieces = list(construction.pieces)
                changed_pieces[piece_index] = Piece(PieceType.GATE, piece.length)
                candidate = Construction(
                    construction.start,
                    construction.start_heading,
                    changed_pieces,
                    list(construction.connectors),
                )
                checked = self._validate(candidate)
                if checked and checked.second_gate_bonus:
                    self.stats.second_gate_upgrades += 1
                    return candidate, checked
        return construction, result

    def _signature(self, construction, result):
        """Ignore the starting index while preserving every piece and turn.

        Inventory totals alone lose gate locations and collinear piece splits,
        both of which affect later replacements. Pair each vertex with its
        outgoing piece and connector before choosing a canonical rotation.
        """
        boundary = tuple(
            (
                round(x, 5),
                round(y, 5),
                piece.piece_type.value,
                piece.length,
                connector.connector_type.value,
                connector.reflex,
            )
            for (x, y), piece, connector in zip(
                result.vertices,
                construction.pieces,
                construction.connectors,
                strict=True,
            )
        )
        return min(
            boundary[index:] + boundary[:index] for index in range(len(boundary))
        )

    def _face_options(self, deadline):
        """Enumerate stock-feasible straight faces of length at most 30."""
        counts = available_pieces(self.inventory)
        choices = sorted(counts, key=lambda piece: (piece.length, piece.is_gate))
        options = {}
        used = Counter()
        # Six minimum-length pieces fill a face. Each extra piece needs one
        # straight connector; corner connectors are handled during packing.
        max_pieces = min(
            MAX_FACE_LENGTH // MIN_WALL_LENGTH,
            self.inventory.connectors.get(STRAIGHT, 0) + 1,
        )

        def extend_face(first_choice, length, pieces):
            if time.process_time() >= deadline:
                return
            if pieces:
                options.setdefault(length, []).append(tuple(pieces))
            if len(pieces) == max_pieces:
                return
            for choice_index in range(first_choice, len(choices)):
                if time.process_time() >= deadline:
                    return
                piece = choices[choice_index]
                if length + piece.length > MAX_FACE_LENGTH:
                    break
                if used[piece] == counts[piece]:
                    continue
                used[piece] += 1
                extend_face(choice_index, length + piece.length, pieces + [piece])
                used[piece] -= 1

        extend_face(0, 0, [])
        for variants in options.values():
            variants.sort(
                key=lambda face: (
                    len(face),
                    sum(piece.is_gate for piece in face),
                    tuple((piece.length, piece.is_gate) for piece in face),
                )
            )
        return options

    def _pack_faces(self, template, options, deadline):
        """Find the first inventory-feasible packing of a seed template."""
        stock = self.inventory.copy()
        for turn in template.corner_turns:
            if not stock.take_connector(TURN[turn].connector_type):
                return None
        chosen_faces = []
        failed_states = set()

        def pack_next_face(face_index, remaining, has_gate):
            if time.process_time() >= deadline:
                return None
            if face_index == len(template.face_lengths):
                if not has_gate:
                    return None
                pieces, connectors = [], []
                for face, turn in zip(chosen_faces, template.corner_turns, strict=True):
                    pieces.extend(face)
                    connectors.extend([TURN[turn]] + [TURN[0]] * (len(face) - 1))
                return Construction((0, 0), 0, pieces, connectors)

            # Different earlier face choices can leave exactly the same
            # remaining packing problem. Remember failures, preserving the
            # order and first successful packing of the original traversal.
            state_key = (
                face_index,
                tuple(remaining.walls.items()),
                tuple(remaining.gates.items()),
                remaining.connectors.get(STRAIGHT, 0),
                has_gate,
            )
            if state_key in failed_states:
                self.stats.packing_states_pruned += 1
                return None

            variants = options.get(template.face_lengths[face_index], [])
            if not has_gate:
                variants = sorted(
                    variants,
                    key=lambda face: (
                        not any(piece.is_gate for piece in face),
                        len(face),
                    ),
                )
            for face in variants:
                if time.process_time() >= deadline:
                    return None
                straight_needed = len(face) - 1
                if remaining.connectors.get(STRAIGHT, 0) < straight_needed:
                    continue
                # A branch owns its stock copy, so a failed packing never
                # needs manual restoration of partially consumed inventory.
                next_stock = remaining.copy()
                if not all(next_stock.take_piece(piece) for piece in face):
                    continue
                next_stock.connectors[STRAIGHT] = (
                    next_stock.connectors.get(STRAIGHT, 0) - straight_needed
                )
                chosen_faces.append(face)
                found = pack_next_face(
                    face_index + 1,
                    next_stock,
                    has_gate or any(piece.is_gate for piece in face),
                )
                chosen_faces.pop()
                if found:
                    return found
            if time.process_time() < deadline:
                failed_states.add(state_key)
            return None

        return pack_next_face(0, stock, False)

    def _prepare_placements(self):
        """Compute room-dependent placement targets once per search."""
        room = self.room.polygon
        usable_area = room.buffer(TOL)
        self._prepared_room = prep(usable_area)
        self._room_bounds = usable_area.bounds
        # TOL is a distance. Area bounds must use the same buffered room as
        # the official containment check, rather than adding TOL to room.area.
        self._usable_room_area = usable_area.area
        self._room_corners = self.room.get_boundary_points()
        angles = []
        for start, end in zip(
            self._room_corners,
            self._room_corners[1:] + self._room_corners[:1],
            strict=True,
        ):
            angles.append(
                math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 180
            )
        angles.extend(range(0, 180, 15))
        self._placement_angles = []
        for angle in angles:
            if all(abs(angle - other) > 1e-6 for other in self._placement_angles):
                self._placement_angles.append(angle)

        centers = []
        if room.covers(room.centroid):
            centers.append((room.centroid.x, room.centroid.y))
        interior = room.representative_point()
        centers.append((interior.x, interior.y))
        min_x, min_y, max_x, max_y = room.bounds
        for column in range(8):
            for row in range(8):
                point = (
                    min_x + (max_x - min_x) * column / 7,
                    min_y + (max_y - min_y) * row / 7,
                )
                if room.contains(Point(point)):
                    centers.append(point)
        self._placement_centers = list(dict.fromkeys(centers))

    def find_placements(self, shape, deadline):
        """Try room headings, an interior grid, and exact corner alignments."""
        vertices, _, _ = build_geometry(shape)
        polygon = Polygon(vertices)
        if not polygon.is_valid or polygon.area > self._usable_room_area + 1e-8:
            return
        center = polygon.centroid
        offsets = [(x - center.x, y - center.y) for x, y in vertices]
        tried = set()
        accepted = 0
        room_min_x, room_min_y, room_max_x, room_max_y = self._room_bounds
        for angle in self._placement_angles:
            radians = math.radians(angle)
            cosine, sine = math.cos(radians), math.sin(radians)
            rotated = [
                (x * cosine - y * sine, x * sine + y * cosine) for x, y in offsets
            ]
            offset_xs, offset_ys = zip(*rotated, strict=True)
            # Every vertex must fit inside the room's bounding box. These
            # intervals reject impossible headings and centers before GEOS.
            left = room_min_x - min(offset_xs)
            right = room_max_x - max(offset_xs)
            bottom = room_min_y - min(offset_ys)
            top = room_max_y - max(offset_ys)
            if left > right + 1e-8 or bottom > top + 1e-8:
                self.stats.heading_bounds_pruned += 1
                continue
            anchors = self._placement_centers + [
                (x - offset_x, y - offset_y)
                for x, y in self._room_corners
                for offset_x, offset_y in rotated
            ]
            for x, y in anchors:
                if time.process_time() >= deadline:
                    return
                if not (
                    left - 1e-8 <= x <= right + 1e-8
                    and bottom - 1e-8 <= y <= top + 1e-8
                ):
                    self.stats.placement_bounds_pruned += 1
                    continue
                placement = (round(x, 5), round(y, 5), round(angle, 5))
                if placement in tried:
                    continue
                tried.add(placement)
                self.stats.seed_attempts += 1
                points = [
                    (x + offset_x, y + offset_y) for offset_x, offset_y in rotated
                ]
                if not self._prepared_room.covers(Polygon(points)):
                    self.stats.geometry_pruned += 1
                    continue
                candidate = Construction(
                    points[0], angle, list(shape.pieces), list(shape.connectors)
                )
                result = self._validate(candidate)
                if result:
                    self.stats.seeds_validated += 1
                    yield self.try_second_gate(candidate, result)
                    accepted += 1
                    if accepted == 2:
                        return  # Leave room for seeds in other parts of the room.

    def _seed_templates(self, options, deadline):
        sizes = sorted(options, reverse=True)
        templates = []
        right_count = self.inventory.connectors.get(RIGHT, 0)
        diagonal_count = self.inventory.connectors.get(DIAGONAL, 0)
        if right_count >= 4:
            for width in sizes:
                for height in sizes:
                    if height <= width:
                        templates.append(
                            SeedTemplate(width * height, (width, height) * 2, (2,) * 4)
                        )
        # Symmetric diagonal shapes need only three independently chosen sizes.
        diagonal_sizes = (
            sizes
            if diagonal_count >= 8 or (right_count >= 2 and diagonal_count >= 4)
            else []
        )
        for horizontal, diagonal, vertical in itertools.product(
            diagonal_sizes, repeat=3
        ):
            if time.process_time() >= deadline:
                break
            if right_count >= 2 and diagonal_count >= 4:
                area = horizontal * vertical + (
                    horizontal + vertical
                ) * diagonal / math.sqrt(2)
                if area <= self._usable_room_area + 1e-8:
                    templates.append(
                        SeedTemplate(
                            area,
                            (horizontal, diagonal, vertical) * 2,
                            (2, 1, 1) * 2,
                        )
                    )
            if diagonal_count >= 8:
                area = (
                    horizontal * vertical
                    + math.sqrt(2) * diagonal * (horizontal + vertical)
                    + diagonal**2
                )
                if area <= self._usable_room_area + 1e-8:
                    templates.append(
                        SeedTemplate(
                            area,
                            (horizontal, diagonal, vertical, diagonal) * 2,
                            (1,) * 8,
                        )
                    )
        return sorted(
            templates,
            key=lambda template: (
                -template.area,
                len(template.face_lengths),
                template.face_lengths,
            ),
        )

    def _place_templates(
        self, templates, options, attempted_shapes, deadline, *, first_fit
    ):
        for template in templates:
            if time.process_time() >= deadline:
                return
            # Even perfect placement cannot beat this geometry score plus the
            # possible gate bonus. Skip templates below the retained seed floor.
            upper_score = (
                BASELINE_SCORE
                + self.weights.A * template.area
                + self.weights.C * sum(template.face_lengths)
                + self._possible_gate_bonus
            )
            if upper_score < self._seed_score_floor - 1e-8:
                self.stats.seed_bound_pruned += 1
                continue
            shape = self._pack_faces(template, options, deadline)
            if shape is None:
                continue
            shape_key = (tuple(shape.pieces), tuple(shape.connectors))
            if shape_key not in attempted_shapes:
                attempted_shapes[shape_key] = False
                for candidate, result in self.find_placements(shape, deadline):
                    attempted_shapes[shape_key] = True
                    yield candidate, result
            if first_fit and attempted_shapes[shape_key]:
                return

    def generate_seeds(self, deadline):
        # Try inexpensive single-piece faces before enumerating composites.
        single_faces = {}
        for piece in sorted(
            available_pieces(self.inventory),
            key=lambda piece: (piece.length, piece.is_gate),
        ):
            single_faces.setdefault(piece.length, []).append((piece,))
        simple_templates = self._seed_templates(single_faces, deadline)
        rectangles = [
            template for template in simple_templates if len(template.face_lengths) == 4
        ]
        diagonal_shapes = [
            template for template in simple_templates if len(template.face_lengths) != 4
        ]
        attempted_shapes = {}

        # Keep both a large fallback and a compact fallback for perimeter-heavy weights.
        yield from self._place_templates(
            rectangles, single_faces, attempted_shapes, deadline, first_fit=True
        )
        yield from self._place_templates(
            reversed(rectangles),
            single_faces,
            attempted_shapes,
            deadline,
            first_fit=True,
        )
        yield from self._place_templates(
            diagonal_shapes, single_faces, attempted_shapes, deadline, first_fit=True
        )

        composite_faces = self._face_options(deadline)
        all_templates = self._seed_templates(composite_faces, deadline)
        composite_rectangles = [
            template for template in all_templates if len(template.face_lengths) == 4
        ]
        yield from self._place_templates(
            composite_rectangles,
            composite_faces,
            attempted_shapes,
            deadline,
            first_fit=True,
        )

        # Refine seeds in objective order after the quick fallbacks are safe.
        all_templates.sort(
            key=lambda template: (
                -(
                    self.weights.A * template.area
                    + self.weights.C * sum(template.face_lengths)
                )
            )
        )
        yield from self._place_templates(
            all_templates, composite_faces, attempted_shapes, deadline, first_fit=False
        )

    def _patterns(self, path_size, initial_turn_steps, target_heading_steps, stock):
        """Return (path turns, turn at B, relative piece headings).

        All angles here are integer multiples of 45 degrees. Path turns belong
        to A and the internal vertices. Headings are relative to the first
        removed piece; target_heading_steps is the unchanged heading at B.
        Construction.start_heading and headings() instead use degrees.
        """
        cache_key = (
            path_size,
            initial_turn_steps,
            target_heading_steps,
            tuple(
                min(stock.connectors.get(kind, 0), path_size + 1)
                for kind in (STRAIGHT, RIGHT, DIAGONAL)
            ),
        )
        if cache_key in self._pattern_cache:
            return self._pattern_cache[cache_key]
        available_turns = tuple(
            turn
            for turn in (0, -2, 2, -1, 1)
            if stock.connectors.get(TURN[turn].connector_type, 0) > 0
        )
        if path_size >= 5:
            # Five pieces form an outward rectangular bump. Six split its middle
            # segment; arbitrary longer patterns would multiply the free lengths.
            relative_heading_steps = (
                [0, -2, 0, 2, 0] if path_size == 5 else [0, -2, 0, 0, 2, 0]
            )
            patterns = [
                (
                    initial_turn_steps,
                    *(
                        end - start
                        for start, end in itertools.pairwise(relative_heading_steps)
                    ),
                )
            ]
        else:
            patterns = itertools.product(available_turns, repeat=path_size)

        feasible = []
        for turn_steps in patterns:
            if any(turn not in available_turns for turn in turn_steps):
                continue
            last_heading_steps = -initial_turn_steps + sum(turn_steps)
            terminal_turn_steps = target_heading_steps - last_heading_steps
            if terminal_turn_steps not in available_turns:
                continue
            needed = Counter(
                TURN[turn].connector_type for turn in (*turn_steps, terminal_turn_steps)
            )
            if any(
                count > stock.connectors.get(kind, 0) for kind, count in needed.items()
            ):
                continue
            relative_heading_steps = []
            heading_steps = -initial_turn_steps
            for turn in turn_steps:
                heading_steps += turn
                relative_heading_steps.append(heading_steps)
            if (relative_heading_steps[-1] - relative_heading_steps[-2]) % 4 == 0:
                continue
            feasible.append((turn_steps, terminal_turn_steps, relative_heading_steps))
        self._pattern_cache[cache_key] = feasible
        return feasible

    def _replacement_paths(self, start, end, directions, counts, deadline):
        """Solve closure and check length stock before doing geometry work.

        Walls and gates of the same length share a path geometry. Enumerate
        lengths once; assign piece types only after the cheap checks pass.
        """
        length_counts = Counter()
        for piece, count in counts.items():
            length_counts[piece.length] += count
        choices = sorted(length_counts, reverse=True)
        path_size = len(directions)
        for prefix in itertools.product(choices, repeat=path_size - 2):
            if time.process_time() >= deadline:
                return
            self.stats.local_candidates += 1
            self.stats.candidates_by_path[path_size] = (
                self.stats.candidates_by_path.get(path_size, 0) + 1
            )
            points = [start]
            # Only the first path_size-2 directions have freely chosen lengths.
            for length, (direction_x, direction_y) in zip(
                prefix, directions, strict=False
            ):
                points.append(
                    (
                        points[-1][0] + length * direction_x,
                        points[-1][1] + length * direction_y,
                    )
                )
            displacement = (end[0] - points[-1][0], end[1] - points[-1][1])
            solved = solve_last_two(displacement, directions[-2], directions[-1])
            if solved is None:
                continue
            last_lengths = [round(length) for length in solved]
            if any(
                rounded < MIN_WALL_LENGTH
                or abs(length - rounded) > TOL
                or rounded not in length_counts
                for length, rounded in zip(solved, last_lengths, strict=True)
            ):
                self.stats.inventory_pruned += 1
                continue
            lengths = (*prefix, *last_lengths)
            if any(
                needed > length_counts[length]
                for length, needed in Counter(lengths).items()
            ):
                self.stats.inventory_pruned += 1
                continue
            direction_x, direction_y = directions[-2]
            middle = (
                points[-1][0] + last_lengths[0] * direction_x,
                points[-1][1] + last_lengths[0] * direction_y,
            )
            yield lengths, [*points, middle, end]

    def _piece_assignments(self, lengths, counts, *, needs_gate):
        """Assign walls/gates to a path without exceeding either stock."""
        by_length = {}
        for piece in counts:
            by_length.setdefault(piece.length, []).append(piece)
        for pieces in itertools.product(*(by_length[length] for length in lengths)):
            if needs_gate and not any(piece.is_gate for piece in pieces):
                continue
            if any(needed > counts[piece] for piece, needed in Counter(pieces).items()):
                self.stats.inventory_pruned += 1
                continue
            yield list(pieces)

    def generate_local_replacements(self, construction, result, deadline):
        """Improve one or two adjacent pieces while keeping the rest fixed."""
        vertices = result.vertices
        piece_headings = headings(construction)
        remaining = self._remaining(construction)
        parent_score = score_construction(result, self.weights)
        parent_signed_area = cross_sum(vertices + vertices[:1]) / 2
        # Optional rectangular bumps run before general three/four-piece paths.
        path_sizes = [size for size in (5, 6, 3, 4) if size <= self.local_path_cap]

        for removed in (1, 2):
            if len(construction.pieces) - removed < 2:
                continue
            for edge in range(len(construction.pieces)):
                if time.process_time() >= deadline:
                    return
                old_pieces = construction.pieces[edge:] + construction.pieces[:edge]
                old_connectors = (
                    construction.connectors[edge:] + construction.connectors[:edge]
                )
                old_vertices = vertices[edge:] + vertices[:edge]
                start, end = old_vertices[0], old_vertices[removed]
                stock = remaining.copy()
                for piece in old_pieces[:removed]:
                    pool = stock.gates if piece.is_gate else stock.walls
                    pool[piece.length] += 1
                # Release both endpoint connectors as well as the internal ones.
                for connector in old_connectors[: removed + 1]:
                    stock.connectors[connector.connector_type] += 1
                counts = available_pieces(stock)
                initial_turn_steps = round(old_connectors[0].turn_angle() / 45)
                target_heading_steps = round(
                    sum(
                        connector.turn_angle()
                        for connector in old_connectors[1 : removed + 1]
                    )
                    / 45
                )
                removed_cross = cross_sum(old_vertices[: removed + 1])
                removed_length = sum(piece.length for piece in old_pieces[:removed])
                needs_gate = not any(piece.is_gate for piece in old_pieces[removed:])

                for path_size in path_sizes:
                    if path_size >= 5 and (
                        removed != 1 or removed_length < 3 * MIN_WALL_LENGTH
                    ):
                        continue
                    for (
                        turn_steps,
                        terminal_turn_steps,
                        relative_heading_steps,
                    ) in self._patterns(
                        path_size, initial_turn_steps, target_heading_steps, stock
                    ):
                        if time.process_time() >= deadline:
                            return
                        self.stats.patterns += 1
                        directions = [
                            (
                                math.cos(
                                    math.radians(
                                        piece_headings[edge] + heading_steps * 45
                                    )
                                ),
                                math.sin(
                                    math.radians(
                                        piece_headings[edge] + heading_steps * 45
                                    )
                                ),
                            )
                            for heading_steps in relative_heading_steps
                        ]
                        for lengths, path in self._replacement_paths(
                            start, end, directions, counts, deadline
                        ):
                            area = abs(
                                parent_signed_area
                                + (cross_sum(path) - removed_cross) / 2
                            )
                            perimeter = result.perimeter - removed_length + sum(lengths)
                            upper_score = (
                                BASELINE_SCORE
                                + self.weights.A * area
                                + self.weights.C * perimeter
                                + self._possible_gate_bonus
                            )
                            if upper_score <= parent_score + 1e-8:
                                continue
                            # Gate requirements and face lengths are also cheap.
                            assignments = self._piece_assignments(
                                lengths, counts, needs_gate=needs_gate
                            )
                            first_pieces = next(assignments, None)
                            if first_pieces is None:
                                continue
                            first_candidate = splice(
                                construction,
                                edge,
                                removed,
                                first_pieces,
                                [TURN[turn] for turn in turn_steps],
                                TURN[terminal_turn_steps],
                            )
                            if any(
                                face.length > MAX_FACE_LENGTH + TOL
                                for face in compute_faces(first_candidate)
                            ):
                                self.stats.geometry_pruned += 1
                                continue
                            polygon = Polygon(path[:-1] + old_vertices[removed:])
                            if not polygon.is_valid or not self._prepared_room.covers(
                                polygon
                            ):
                                self.stats.geometry_pruned += 1
                                continue
                            # Piece types do not change geometry. Check it once,
                            # then validate each stock-feasible gate arrangement.
                            for pieces in itertools.chain([first_pieces], assignments):
                                if time.process_time() >= deadline:
                                    return
                                candidate = Construction(
                                    first_candidate.start,
                                    first_candidate.start_heading,
                                    pieces + old_pieces[removed:],
                                    list(first_candidate.connectors),
                                )
                                checked = self._validate(candidate)
                                if checked is None:
                                    continue
                                candidate, checked = self.try_second_gate(
                                    candidate, checked
                                )
                                if (
                                    score_construction(checked, self.weights)
                                    > parent_score + 1e-8
                                ):
                                    self.stats.valid_children += 1
                                    self.stats.valid_by_path[path_size] = (
                                        self.stats.valid_by_path.get(path_size, 0) + 1
                                    )
                                    yield candidate, checked

    def build_enclosure(self):
        started = time.process_time()
        budget = self.cpu_budget
        # Read the judge's timer without changing it or catching its interrupt.
        if hasattr(signal, "getitimer") and hasattr(signal, "ITIMER_PROF"):
            remaining, _ = signal.getitimer(signal.ITIMER_PROF)
            if remaining > 0:
                budget = min(budget, 0.8 * remaining)
        deadline = started + budget
        self.stats = SearchStats()
        self.best_score = float("-inf")
        self._pattern_cache = {}
        self._prepare_placements()
        self._seed_score_floor = float("-inf")
        best = None
        frontier = []
        considered = set()

        def retain(construction, result, depth):
            nonlocal best
            score = score_construction(result, self.weights)
            if score > self.best_score:
                best, self.best_score = construction, score
                if self.stats.first_valid_cpu is None:
                    self.stats.first_valid_cpu = time.process_time() - started
            if self.seeds_only:
                self._seed_score_floor = self.best_score
            self.stats.best_by_depth[depth] = max(
                score, self.stats.best_by_depth.get(depth, float("-inf"))
            )
            signature = self._signature(construction, result)
            if signature in considered:
                return
            considered.add(signature)
            state = SearchState(construction, result, score, depth)
            frontier.append(state)
            # The frontier is tiny. A sorted list makes both ends explicit and
            # preserves first-in order when scores tie.
            frontier.sort(key=lambda entry: entry.score, reverse=True)
            if len(frontier) > self.frontier_width:
                discarded = frontier.pop()
                if discarded is state:
                    return
            self.stats.unique_states_retained += 1
            if not self.seeds_only and len(frontier) == self.frontier_width:
                self._seed_score_floor = frontier[-1].score

        if self.inventory.has_gate():
            seed_deadline = (
                deadline
                if self.seeds_only
                else min(deadline, started + min(6.0, budget * 0.35))
            )
            for construction, result in self.generate_seeds(seed_deadline):
                retain(construction, result, 0)
            self.stats.seed_cpu = time.process_time() - started
            if not self.seeds_only:
                while frontier and time.process_time() < deadline:
                    parent = frontier.pop(0)
                    self.stats.expansions += 1
                    # Limit each parent to 1/16 of a short budget, at most 2s.
                    slice_deadline = min(
                        deadline, time.process_time() + max(0.02, min(2.0, budget / 16))
                    )
                    for child, checked in self.generate_local_replacements(
                        parent.construction, parent.result, slice_deadline
                    ):
                        retain(child, checked, parent.depth + 1)
        self.stats.cpu = time.process_time() - started
        self.note = (
            f"Anytime seeds/local search: score {self.best_score:g}; "
            f"{self.stats.seeds_validated} seeds, {self.stats.local_candidates} local candidates, "
            f"{self.stats.valid_children} improving children, {self.stats.cpu:.3f}s CPU"
        )
        return best
