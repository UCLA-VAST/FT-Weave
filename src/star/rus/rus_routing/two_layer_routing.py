from collections import defaultdict, deque

from src.ds import FactoryPool, TFactoryPool


def create_flow_network() -> tuple[dict, str, str]:
    """Create an empty flow network for bipartite matching.

    Returns (graph, source, sink).
    """
    graph = defaultdict(lambda: defaultdict(int))
    source = "S"
    sink = "T"
    return graph, source, sink


def add_edge(graph: dict, u: str, v: str, capacity: int) -> None:
    """Add edge with given capacity to the graph."""
    graph[u][v] += capacity


def bfs(graph: dict, source: str, sink: str, parent: dict) -> bool:
    """BFS to find augmenting path in residual graph."""
    visited = {source}
    queue = deque([source])

    while queue:
        u = queue.popleft()

        for v in graph[u]:
            if v not in visited and graph[u][v] > 0:
                visited.add(v)
                queue.append(v)
                parent[v] = u
                if v == sink:
                    return True
    return False


def ford_fulkerson(
    graph: dict, source: str, sink: str, n: int
) -> tuple[int, list[tuple[int, int]]]:
    """Run Ford-Fulkerson (Edmonds-Karp) on `graph` and extract matching.

    Returns (max_flow, matching_edges) where matching_edges are pairs (u,v)
    from left partition index u to right partition index v.
    """
    parent = {}
    max_flow = 0
    flow_edges = []

    # Find augmenting paths using BFS
    while bfs(graph, source, sink, parent):
        # Find minimum capacity along the path
        path_flow = 10**9
        s = sink

        while s != source:
            path_flow = min(path_flow, graph[parent[s]][s])
            s = parent[s]

        # Update residual capacities
        v = sink
        while v != source:
            u = parent[v]
            graph[u][v] -= path_flow
            graph[v][u] += path_flow
            v = parent[v]

        max_flow += path_flow
        parent = {}

    # Extract matching edges (flow from left to right partition)
    for u in range(n):
        left_node = f"L{u}"
        for v in range(n):
            right_node = f"R{v}"
            # Check if there's flow (reverse edge has capacity)
            if graph[right_node][left_node] > 0:
                flow_edges.append((u, v))

    return int(max_flow), flow_edges


def chain_decomposition_matching(
    perm1: list[int], perm2: list[int]
) -> tuple[list[tuple[int, int]], list[list[int]]]:
    """
    Given two permutations, find minimum chain decomposition using Dilworth's theorem.
    Uses Ford-Fulkerson algorithm for maximum bipartite matching.

    Args:
        perm1: First permutation (list where perm1[i] is the value at position i)
        perm2: Second permutation (list where perm2[i] is the value at position i)

    Returns:
        matching: List of tuples (u, v) representing edges in the matching
        chains: List of chains in the minimum chain decomposition
    """
    n = len(perm1)

    # Create inverse permutations to find positions
    inv_perm1 = [0] * n
    inv_perm2 = [0] * n
    for i in range(n):
        inv_perm1[perm1[i]] = i
        inv_perm2[perm2[i]] = i

    # Define partial order: element i < element j if
    # position of i in perm1 < position of j in perm1 AND
    # position of i in perm2 < position of j in perm2
    def is_less_than(i, j):
        return inv_perm1[i] < inv_perm1[j] and inv_perm2[i] < inv_perm2[j]

    # Build bipartite graph for maximum matching using module-level helpers
    # Left partition: elements as potential predecessors
    # Right partition: elements as potential successors
    # Edge (u, v) exists if u < v in the partial order
    graph, source, sink = create_flow_network()

    # Add source to left partition edges (capacity 1)
    for u in range(n):
        add_edge(graph, source, f"L{u}", 1)

    # Add right partition to sink edges (capacity 1)
    for v in range(n):
        add_edge(graph, f"R{v}", sink, 1)

    # Add edges for comparable elements
    for u in range(n):
        for v in range(n):
            if is_less_than(u, v):
                add_edge(graph, f"L{u}", f"R{v}", 1)

    # Run Ford-Fulkerson to find maximum matching
    max_flow, matching = ford_fulkerson(graph, source, sink, n)

    # Build chains from the matching
    # Create adjacency list from matching
    next_in_chain = {}
    for u, v in matching:
        next_in_chain[u] = v

    used = [False] * n
    chains = []

    # Find chain starts (elements not matched on the right side)
    matched_right = {v for u, v in matching}

    for u in range(n):
        if not used[u] and u not in matched_right:
            # Build chain starting from u
            chain = []
            curr = u
            while curr is not None and not used[curr]:
                chain.append(curr)
                used[curr] = True
                curr = next_in_chain.get(curr)
            chains.append(chain)

    # Add singleton chains for remaining elements
    for u in range(n):
        if not used[u]:
            chains.append([u])
            used[u] = True

    return matching, chains


def two_layer_routing(
    factory_pool: FactoryPool | TFactoryPool,
    logic_qubit_locations: list[tuple[int, int]],
    qubit_factory_pairs: list[tuple[int, int]],
) -> list[list[tuple[int, int, int, int, int, int]]]:
    """
    Two layer routing
    """
    # split factories according to rows
    compatible_row_movement_to_factories = defaultdict(list)
    for qubit, factory_id in qubit_factory_pairs:
        factory = factory_pool.get_factory_by_id(factory_id=factory_id)
        x_f, y_f = factory.location
        x_q, y_q = logic_qubit_locations[qubit]
        compatible_row_movement_to_factories[(y_f, y_q)].append(
            (factory_id, x_f, qubit, x_q)
        )

    routing_batches = []
    movement_vectors = dict()
    # sort factory based on locations and solve two-layer routing
    for y_pair in compatible_row_movement_to_factories.keys():
        sorted_factories = sorted(
            compatible_row_movement_to_factories[y_pair], key=lambda x: x[1]
        )
        sorted_qubit_indices = sorted(
            range(len(sorted_factories)), key=lambda i: sorted_factories[i][3]
        )
        factory_list = list(range(len(sorted_factories)))
        matching, chains = chain_decomposition_matching(
            factory_list, sorted_qubit_indices
        )
        for chain in chains:
            batch = []
            vec = []
            for idx in chain:
                qubit = sorted_factories[idx][2]
                factory_id = sorted_factories[idx][0]
                x_q, y_q = logic_qubit_locations[qubit]
                factory = factory_pool.get_factory_by_id(factory_id=factory_id)
                x_f, y_f = factory.location
                batch.append((qubit, x_q, y_q, factory_id, x_f, y_f))
                vec.append(sorted_factories[idx][1])
                vec.append(sorted_factories[idx][3])
            vec = tuple(vec)
            # merge movement from two rows if they have the same pattern
            if vec in movement_vectors:
                batch_id = movement_vectors[vec]
                routing_batches[batch_id] += batch
            else:
                movement_vectors[vec] = len(routing_batches)
                routing_batches.append(batch)

    return routing_batches


def verify_chain_decomposition(
    perm1: list[int], perm2: list[int], chains: list[list[int]]
) -> tuple[bool, str]:
    """
    Verify that the chain decomposition is valid.
    """
    n = len(perm1)

    # Create inverse permutations
    inv_perm1 = [0] * n
    inv_perm2 = [0] * n
    for i in range(n):
        inv_perm1[perm1[i]] = i
        inv_perm2[perm2[i]] = i

    # Check all elements are covered exactly once
    covered = set()
    for chain in chains:
        for elem in chain:
            if elem in covered:
                return False, f"Element {elem} appears in multiple chains"
            covered.add(elem)

    if len(covered) != n:
        return False, f"Not all elements covered: {len(covered)}/{n}"

    # Check each chain is valid
    for chain in chains:
        for i in range(len(chain) - 1):
            u, v = chain[i], chain[i + 1]
            if not (inv_perm1[u] < inv_perm1[v] and inv_perm2[u] < inv_perm2[v]):
                return False, f"Chain {chain} invalid: {u} not less than {v}"

    return True, f"Valid decomposition with {len(chains)} chains"
