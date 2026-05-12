from __future__ import annotations

from babel.env.dynamics import DemandEvent
from babel.env.graph import DEMAND, DEPOT, TRANSIT, GraphEdge, ResourceGraph
from scripts.diagnose_solo_completability import strict_solo_completability


def _test_graph() -> ResourceGraph:
    node_types = (
        DEPOT,
        TRANSIT,
        DEMAND,
        DEPOT,
        DEPOT,
        DEMAND,
        DEMAND,
        DEMAND,
        DEMAND,
        DEMAND,
        TRANSIT,
        TRANSIT,
    )
    edges = (
        GraphEdge(0, 1, 3),
        GraphEdge(1, 2, 3),
        GraphEdge(0, 2, 3),
        GraphEdge(0, 3, 20),
        GraphEdge(0, 4, 20),
        GraphEdge(2, 5, 1),
        GraphEdge(5, 6, 1),
        GraphEdge(6, 7, 1),
        GraphEdge(7, 8, 1),
        GraphEdge(8, 9, 1),
        GraphEdge(9, 10, 1),
        GraphEdge(10, 11, 1),
        GraphEdge(3, 4, 20),
        GraphEdge(4, 11, 20),
    )
    return ResourceGraph(
        seed=0,
        positions=tuple((0.0, 0.0) for _ in range(12)),
        node_types=node_types,
        edges=edges,
    )


def test_strict_metric_counts_start_to_depot_and_service_steps() -> None:
    graph = _test_graph()
    depot_inventory = {0: [1, 0], 3: [0, 0], 4: [0, 0]}
    starts = [1, 10, 10, 10]

    feasible = strict_solo_completability(
        graph=graph,
        agent_start_nodes=starts,
        depot_inventory=depot_inventory,
        demands=[
            DemandEvent(
                demand_id=0,
                node_id=2,
                resource_type=0,
                deadline=8,
                reward_value=1,
            )
        ],
    )
    infeasible = strict_solo_completability(
        graph=graph,
        agent_start_nodes=starts,
        depot_inventory=depot_inventory,
        demands=[
            DemandEvent(
                demand_id=0,
                node_id=2,
                resource_type=0,
                deadline=7,
                reward_value=1,
            )
        ],
    )

    assert feasible.strict_solo_fraction == 1.0
    assert infeasible.strict_solo_fraction == 0.0


def test_strict_metric_accounts_for_shared_depot_inventory_contention() -> None:
    graph = _test_graph()
    depot_inventory = {0: [1, 0], 3: [0, 0], 4: [0, 0]}

    result = strict_solo_completability(
        graph=graph,
        agent_start_nodes=[0, 0, 0, 0],
        depot_inventory=depot_inventory,
        demands=[
            DemandEvent(
                demand_id=0,
                node_id=2,
                resource_type=0,
                deadline=8,
                reward_value=1,
            ),
            DemandEvent(
                demand_id=1,
                node_id=5,
                resource_type=0,
                deadline=9,
                reward_value=1,
            ),
        ],
    )

    assert result.strict_solo_completed == 1
    assert result.strict_solo_fraction == 0.5
