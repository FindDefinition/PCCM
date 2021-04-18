import json
from typing import Dict, Generic, List, Optional, Set, Tuple, TypeVar, Union


class NamedIO(object):
    def __init__(self, node: Union[str, "Node"], io_name: Optional[str]):
        self.node = node
        self.io_name = io_name

    def to_dict(self):
        return {
            "key": self.key,
            "io_name": self.io_name,
        }

    @property
    def key(self) -> str:
        key = self.node
        if isinstance(self.node, Node):
            key = self.node.key
        return key

    def __repr__(self):
        return "NamedIO[{}, {}]".format(self.key, self.io_name)


class Node(object):
    def __init__(self, key: Optional[str] = None):
        """if key is None, it should be inited after __init__
        in subclasses.
        """
        self.inputs = []  # type: List[NamedIO]
        self.outputs = []  # type: List[NamedIO]
        self._key = key

    @property
    def key(self):
        assert self._key is not None, "you must init key before use Node"
        return self._key

    @key.setter
    def key(self, val):
        self._key = val

    def __hash__(self):
        assert self._key is not None
        return hash(self._key)

    def to_dict(self):
        assert self._key is not None
        return {
            "key": self._key,
            "inputs": [n.to_dict() for n in self.inputs],
            "outputs": [n.to_dict() for n in self.outputs],
        }

    def __repr__(self):
        return json.dumps(self.to_dict(), indent=2)

    def add_input(self, key, io_name=""):
        return self.inputs.append(NamedIO(key, io_name))

    def add_output(self, key, io_name):
        self.outputs.append(NamedIO(key, io_name))

    def clear_output(self):
        self.outputs.clear()


T = TypeVar('T')


def postorder_traversal(node: Node, node_map: Dict[str, Node]):
    stack = [node]
    visited = set()
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        next_nodes = []
        ready = True
        for namedio in node.inputs:
            if namedio.key not in node_map:
                continue
            inp = node_map[namedio.key]
            if not inp in visited:
                next_nodes.append(inp)
                ready = False
        if ready:
            yield node
            visited.add(node)
        else:
            stack.append(node)
            stack.extend(next_nodes)


def _cycle_detection(node_map: Dict[str, Node], node: Node, visited: Set[str],
                     trace: Set[str]):
    visited.add(node.key)
    trace.add(node.key)
    for namedio in node.inputs:
        if namedio.key not in visited:
            if _cycle_detection(node_map, node_map[namedio.key], visited,
                                trace):
                return True
            elif namedio.key in trace:
                return True
    trace.remove(node.key)
    return False


def cycle_detection(node_map: Dict[str, Node]):
    visited = set()
    trace = set()
    for node in node_map.values():
        if node.key not in visited:
            if _cycle_detection(node_map, visited, trace):
                return True
    return False


class Graph(object):
    """directed graph of input-based nodes (nodes that only contains input info).
    """
    def __init__(self, nodes: List[Node]):
        # TODO check graph is valid (all input key exists in nodes)
        self.node_map = {n.key: n for n in nodes}  # type: Dict[str, Node]
        self._has_cycle = cycle_detection(self.node_map)

    def __getitem__(self, key) -> Node:
        return self.node_map[key]

    def __contains__(self, key) -> bool:
        return key in self.node_map

    def has_cycle(self):
        return self._has_cycle

    def nodes(self):
        yield from self.node_map.values()

    def node_keys(self):
        yield from self.node_map.keys()

    def postorder(self, node: Node):
        """can only postorder traversal DAG.
        """
        assert node.key in self.node_map
        assert not self.has_cycle()
        yield from postorder_traversal(node, self.node_map)

    def source_nodes(self):
        for n in self.node_map.values():
            if len(n.inputs) == 0:
                yield n

    def add_node(self, node: Node):
        assert node.key not in self.node_map
        self.node_map[node.key] = node
        # update has cycle
        self._has_cycle = cycle_detection(self.node_map)

    def is_source_of(self, lfs: Node, rfs: Node):
        """check whether lfs is source of rfs.
        the graph must be DAG.
        """
        assert not self._has_cycle, "graph must be DAG"
        stack = [rfs]
        while stack:
            n = stack.pop()
            if n.key == lfs.key:
                return True
            for io in n.inputs:
                stack.append(self[io.key])
        return False

    def get_sources_of(self, node: Node):
        assert not self._has_cycle, "graph must be DAG"
        all_sources = set()
        stack = []
        for inp in node.inputs:
            if inp.key in self:
                stack.append(self[inp.key])
        while stack:
            n = stack.pop()
            if n in all_sources:
                continue
            yield n
            all_sources.add(n)
            for io in n.inputs:
                if io.key in self:
                    stack.append(self[io.key])

    def is_branch_node(self, node: Node):
        """naive implementation, O(N)
        """
        assert not self._has_cycle, "graph must be DAG"
        all_sources = set(self.get_sources_of(node))
        # 2. iterate remain nodes. if visited nodes contains any node in
        # previous source nodes, the node is branch node.
        visited = set()
        for remain_node in self.nodes():
            if remain_node in all_sources:
                continue
            if remain_node.key == node.key:
                continue
            stack = [remain_node]
            while stack:
                n = stack.pop()
                if n.key in visited:
                    continue
                visited.add(n.key)
                if n.key != node.key:
                    for io in n.inputs:
                        if io.key in self:
                            stack.append(self[io.key])
        for s in all_sources:
            if s.key in visited:
                return True
        return False


def create_node(key: str, *inputs: List[NamedIO]):
    node = Node(key)
    node.inputs = inputs
    return node


if __name__ == "__main__":
    node1 = Node("n1")

    node2 = Node("n2")

    node3 = Node("n3")
    node3.inputs = [
        NamedIO("n1", "inp"),
        NamedIO("n2", "inp"),
        NamedIO("n6", "inp")
    ]

    node4 = Node("n4")
    node5 = Node("n5")
    node5.inputs = [NamedIO("n3", "inp"), NamedIO("n4", "inp")]
    node6 = Node("n6")
    node6.inputs = [NamedIO("n5", "inp")]
    all_nodes = [node1, node2, node3, node4, node5, node6]
    g = Graph(all_nodes)
    print(g.has_cycle())
