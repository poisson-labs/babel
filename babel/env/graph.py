from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from math import hypot
from typing import Final

import numpy as np
from numpy.typing import NDArray

NUM_NODES: Final[int] = 12
DEPOT: Final[str] = "depot"
DEMAND: Final[str] = "demand"
TRANSIT: Final[str] = "transit"
NODE_TYPES: Final[tuple[str, str, str]] = (DEPOT, DEMAND, TRANSIT)


@dataclass(frozen=True, slots=True)
class GraphEdge:
    source: int
    target: int
    cost: int

    def as_tuple(self) -> tuple[int, int, int]:
        return (self.source, self.target, self.cost)


@dataclass(frozen=True, slots=True)
class ResourceGraph:
    seed: int
    positions: tuple[tuple[float, float], ...]
    node_types: tuple[str, ...]
    edges: tuple[GraphEdge, ...]

    def neighbors(self, node_id: int) -> tuple[int, ...]:
        return tuple(
            edge.target if edge.source == node_id else edge.source
            for edge in self.edges
            if edge.source == node_id or edge.target == node_id
        )

    def edge_cost(self, source: int, target: int) -> int:
        for edge in self.edges:
            if {edge.source, edge.target} == {source, target}:
                return edge.cost
        raise KeyError(f"No edge between {source} and {target}")

    def depots(self) -> tuple[int, ...]:
        return tuple(
            node_id for node_id, node_type in enumerate(self.node_types) if node_type == DEPOT
        )

    def demand_nodes(self) -> tuple[int, ...]:
        return tuple(
            node_id for node_id, node_type in enumerate(self.node_types) if node_type == DEMAND
        )

    def shortest_hops(self, start: int) -> dict[int, int]:
        distances: dict[int, int] = {start: 0}
        queue: deque[int] = deque([start])
        while queue:
            current = queue.popleft()
            for neighbor in self.neighbors(current):
                if neighbor not in distances:
                    distances[neighbor] = distances[current] + 1
                    queue.append(neighbor)
        return distances

    def shortest_costs(self, start: int) -> dict[int, int]:
        costs: dict[int, int] = {start: 0}
        pending = set(range(NUM_NODES))
        while pending:
            current = min(pending, key=lambda node_id: costs.get(node_id, 10**9))
            if current not in costs:
                break
            pending.remove(current)
            for neighbor in self.neighbors(current):
                candidate = costs[current] + self.edge_cost(current, neighbor)
                if candidate < costs.get(neighbor, 10**9):
                    costs[neighbor] = candidate
        return costs

    def connectivity_invariant_holds(self, max_hops: int = 4) -> bool:
        depot_distances = [self.shortest_hops(depot) for depot in self.depots()]
        for demand_node in self.demand_nodes():
            if min(distances.get(demand_node, 10**9) for distances in depot_distances) > max_hops:
                return False
        return True

    def to_dict(self) -> dict[str, object]:
        return {
            "seed": self.seed,
            "positions": [[round(x, 8), round(y, 8)] for x, y in self.positions],
            "node_types": list(self.node_types),
            "edges": [list(edge.as_tuple()) for edge in self.edges],
        }


def generate_resource_graph(seed: int, *, max_attempts: int = 10_000) -> ResourceGraph:
    rng = np.random.default_rng(seed)
    for _ in range(max_attempts):
        positions_array = rng.random((NUM_NODES, 2))
        node_types = _sample_node_types(rng)
        undirected_edges = _build_degree_bounded_edges(positions_array)
        if undirected_edges is None:
            continue

        edges = tuple(
            GraphEdge(source=u, target=v, cost=int(rng.integers(1, 4)))
            for u, v in sorted(undirected_edges)
        )
        graph = ResourceGraph(
            seed=seed,
            positions=tuple((float(x), float(y)) for x, y in positions_array),
            node_types=node_types,
            edges=edges,
        )
        if graph.connectivity_invariant_holds(max_hops=4):
            return graph
    raise RuntimeError(f"Could not generate a valid Resource Logistics graph for seed {seed}")


def _sample_node_types(rng: np.random.Generator) -> tuple[str, ...]:
    labels = np.array([DEPOT] * 3 + [DEMAND] * 6 + [TRANSIT] * 3, dtype=object)
    rng.shuffle(labels)
    return tuple(str(label) for label in labels.tolist())


def _build_degree_bounded_edges(
    positions: NDArray[np.float64],
) -> set[tuple[int, int]] | None:
    candidates = _candidate_edges_by_distance(positions)
    edges: set[tuple[int, int]] = set()
    degrees = [0 for _ in range(NUM_NODES)]

    while min(degrees) < 2:
        node_id = min(range(NUM_NODES), key=lambda candidate: degrees[candidate])
        edge = _shortest_available_edge(node_id, candidates, degrees, edges)
        if edge is None:
            return None
        _add_edge(edge, edges, degrees)

    for edge in candidates:
        if _is_connected(edges) and max(degrees) <= 4:
            break
        if edge not in edges and degrees[edge[0]] < 4 and degrees[edge[1]] < 4:
            _add_edge(edge, edges, degrees)

    if not _is_connected(edges):
        return None
    if not all(2 <= degree <= 4 for degree in degrees):
        return None
    return edges


def _candidate_edges_by_distance(positions: NDArray[np.float64]) -> list[tuple[int, int]]:
    candidates: list[tuple[float, int, int]] = []
    for source in range(NUM_NODES):
        for target in range(source + 1, NUM_NODES):
            distance = hypot(
                float(positions[source, 0] - positions[target, 0]),
                float(positions[source, 1] - positions[target, 1]),
            )
            candidates.append((distance, source, target))
    candidates.sort()
    return [(source, target) for _, source, target in candidates]


def _shortest_available_edge(
    node_id: int,
    candidates: list[tuple[int, int]],
    degrees: list[int],
    existing: set[tuple[int, int]],
) -> tuple[int, int] | None:
    for source, target in candidates:
        if node_id not in (source, target):
            continue
        if (source, target) in existing:
            continue
        if degrees[source] < 4 and degrees[target] < 4:
            return (source, target)
    return None


def _add_edge(edge: tuple[int, int], edges: set[tuple[int, int]], degrees: list[int]) -> None:
    source, target = edge
    edges.add(edge)
    degrees[source] += 1
    degrees[target] += 1


def _is_connected(edges: set[tuple[int, int]]) -> bool:
    adjacency: dict[int, list[int]] = {node_id: [] for node_id in range(NUM_NODES)}
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    seen = {0}
    queue: deque[int] = deque([0])
    while queue:
        current = queue.popleft()
        for neighbor in adjacency[current]:
            if neighbor not in seen:
                seen.add(neighbor)
                queue.append(neighbor)
    return len(seen) == NUM_NODES
