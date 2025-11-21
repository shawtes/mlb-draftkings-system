from collections import defaultdict, deque
from typing import Any, Dict, Iterable, List, Set, Union

GraphType = Union[Dict[Any, Iterable[Any]], List[List[Any]]]

def extractCycle(graph: GraphType, start: Any) -> List[Any]:
    
    def get_neighbors(node: Any) -> Iterable[Any]:
        if isinstance(graph, dict):
            return graph.get(node, [])
        if isinstance(node, int) and 0 <= node < len(graph):
            return graph[node]
        return []

    all_nodes: Set[Any] = set()
    if isinstance(graph, dict):
        for u, nbrs in graph.items():
            all_nodes.add(u)
            for v in nbrs:
                all_nodes.add(v)
    else:
        all_nodes.update(range(len(graph)))
        for nbrs in graph:
            for v in nbrs:
                all_nodes.add(v)

    if start not in all_nodes:
        return []

    for nbr in get_neighbors(start):
        if nbr == start:
            return [start, start]

    reverse_adj: Dict[Any, List[Any]] = defaultdict(list)
    for u in all_nodes:
        for v in get_neighbors(u):
            reverse_adj[v].append(u)

    can_reach_start: Set[Any] = set()
    queue = deque([start])
    can_reach_start.add(start)
    while queue:
        v = queue.popleft()
        for p in reverse_adj.get(v, []):
            if p not in can_reach_start:
                can_reach_start.add(p)
                queue.append(p)

    if len(can_reach_start) == 1:
        return []

    path: List[Any] = [start]
    in_path: Set[Any] = {start}

    def dfs(u: Any) -> List[Any]:
        for v in get_neighbors(u):
            if v not in can_reach_start:
                continue
            if v == start:
                return path + [start]
            if v in in_path:
                continue
            in_path.add(v)
            path.append(v)
            res = dfs(v)
            if res:
                return res
            path.pop()
            in_path.remove(v)
        return []

    return dfs(start)

if __name__ == "__main__":
    graph = {
        0: [1],
        1: [2],
        2: [4],
        3: [8],
        4: [0],
    }
    print(extractCycle(graph, 0))