from functools import partial
from typing import Callable, Iterator, Optional, Tuple

from networkx import MultiDiGraph
from networkx.classes.coreviews import AdjacencyView
from relationship_graph.attrs_relations import RelationshipEnum

# utils


def is_empty(some_generator: Iterator) -> bool:
    _exhausted = object()

    return next(some_generator, _exhausted) is _exhausted


# elementary solvers


def find_room_w_type(n: str, room_type: str, graph: MultiDiGraph) -> bool:
    return str(graph.nodes[n].get("room_type")) == room_type


def find_object_w_type(n: str, object_type: str, graph: MultiDiGraph) -> bool:
    return str(graph.nodes[n].get("object_type")) == object_type


def get_out_neighbors_type_w_relation(
    n: str, graph: MultiDiGraph
) -> Iterator[Tuple[str, RelationshipEnum]]:
    uvk2vk: Callable[
        [Tuple[str, str, RelationshipEnum]],
        Tuple[str, RelationshipEnum],
    ] = lambda uvk: (graph.nodes[uvk[1]].get("object_type"), uvk[2])
    return map(uvk2vk, graph.edges(n, keys=True))


def get_neighbors(n: str, graph: MultiDiGraph) -> AdjacencyView:
    return graph.neighbors(n)


def is_object_type(n: str, object_type: str, graph: MultiDiGraph) -> bool:
    return str(graph.nodes[n].get("object_type")) == object_type


def room2objects(
    n: str, object_type: str, graph: MultiDiGraph
) -> AdjacencyView:
    return filter(
        partial(is_object_type, object_type=object_type, graph=graph),
        get_neighbors(n, graph=graph),
    )


def get_node_type(node: dict) -> str:
    return node.get(
        "object_type", node.get("room_type", str(node.get("type")))
    )


def get_node_color(node: dict) -> Optional[str]:
    color = node.get("color", None)
    return color if color != "None" else None


def get_node_material(node: dict) -> Optional[str]:
    material = node.get("material", None)
    return material if material != "None" else None


def edge2uvktype(
    edge: Tuple[str, str, RelationshipEnum], graph: MultiDiGraph
) -> Tuple[str, str, RelationshipEnum]:
    return (
        get_node_type(graph.nodes[edge[0]]),
        get_node_type(graph.nodes[edge[1]]),
        edge[2],
    )
