import copy
import pickle
from typing import Any, Dict, List, Optional, Set, Tuple, cast

import datasets
import networkx
import networkx as nx
import numpy as np
import plotly.graph_objects as go
from ai2thor.controller import Controller
from allenact.utils.misc_utils import unzip
from allenact_plugins.ithor_plugin.ithor_environment import (
    IThorEnvironment,
)
from plotly.offline import iplot
from relationship_graph.attrs_relations import (
    AttributeEnum,
    BrokenEnum,
    MovementEnum,
    NodeTypeEnum,
    OpenableEnum,
    RelationshipEnum,
)
from relationship_graph.category import get_size, load_categories
from relationship_graph.constants import (
    NEAR_SIGMOID_SCALE,
    NEAR_THRESHOLD,
    SPATIAL_SIGMOID_SCALE,
    TOUCH_NEAR_THRESHOLD,
    TOUCH_SIGMOID_SCALE,
    VISIBLE_DISTANCE,
    VISIBLE_SIGMOID_SCALE,
)
from relationship_graph.misc import (
    fast_dist_between_objects_lower_bound,
    fast_dist_to_object_lower_bound,
    get_angle_of_vec_counter_clockwise,
    get_object_radius,
    position_to_floor_tuple,
    position_to_numpy,
    sigmoid_margin_distance,
)
from scipy.spatial.transform import Rotation
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon


class StrictMultiDiGraph(nx.MultiDiGraph):
    def add_edge(self, u_for_edge, v_for_edge, key=None, **attr):
        assert u_for_edge in self, f"{u_for_edge} is not in {self}"
        assert v_for_edge in self, f"{v_for_edge} is not in {self}"
        return super(StrictMultiDiGraph, self).add_edge(
            u_for_edge=u_for_edge,
            v_for_edge=v_for_edge,
            key=key,
            **attr,
        )


def get_object_attributes(
    obj: Dict[str, Any], init_obj: Optional[Dict[str, Any]] = None
) -> Set[AttributeEnum]:
    attr_set = {
        OpenableEnum.from_dict(obj),
        BrokenEnum.from_dict(obj),
        MovementEnum.from_dict(obj, init_obj=init_obj),
    }
    if None in attr_set:
        attr_set.remove(None)

    return cast(Set[AttributeEnum], attr_set)


def get_object_bbox_corners_from_agents_ref_frame(
    agent: Dict[str, Any],
    obj0: Dict[str, Any],
    obj1: Dict[str, Any],
):
    if obj0["objectOrientedBoundingBox"] is not None:
        corners0 = np.array(obj0["objectOrientedBoundingBox"]["cornerPoints"])
    else:
        corners0 = np.array(obj0["axisAlignedBoundingBox"]["cornerPoints"])

    if obj1["objectOrientedBoundingBox"] is not None:
        corners1 = np.array(obj1["objectOrientedBoundingBox"]["cornerPoints"])
    else:
        corners1 = np.array(obj1["axisAlignedBoundingBox"]["cornerPoints"])

    agent_position = np.array(
        [[agent["position"][k] for k in ["x", "y", "z"]]]
    )

    corners0 -= agent_position
    corners1 -= agent_position

    rot_mat = Rotation.from_euler(
        "y", -agent["rotation"]["y"], degrees=True
    ).as_matrix()
    corners0 = np.matmul(rot_mat, corners0.T).T
    corners1 = np.matmul(rot_mat, corners1.T).T

    return corners0, corners1


def add_spatial_relationship_edges(
    graph: nx.MultiDiGraph,
    agent: Dict[str, Any],
    obj0: Dict[str, Any],
    obj1: Dict[str, Any],
):
    if not (obj0["visible"] and obj1["visible"]):
        return

    corners0, corners1 = get_object_bbox_corners_from_agents_ref_frame(
        agent=agent,
        obj0=obj0,
        obj1=obj1,
    )

    xmin0, ymin0, zmin0 = corners0.min(0)
    xmax0, ymax0, zmax0 = corners0.max(0)

    xmin1, ymin1, zmin1 = corners1.min(0)
    xmax1, ymax1, zmax1 = corners1.max(0)

    oid0 = obj0["objectId"]
    oid1 = obj1["objectId"]

    if xmax1 <= xmin0:
        graph.add_edge(oid0, oid1, key=RelationshipEnum.RIGHT)
        graph.add_edge(oid1, oid0, key=RelationshipEnum.LEFT)

    if xmax0 <= xmin1:
        graph.add_edge(oid1, oid0, key=RelationshipEnum.RIGHT)
        graph.add_edge(oid0, oid1, key=RelationshipEnum.LEFT)

    if ymax1 <= ymin0:
        graph.add_edge(oid0, oid1, key=RelationshipEnum.ABOVE)
        graph.add_edge(oid1, oid0, key=RelationshipEnum.BELOW)

    if ymax0 <= ymin1:
        graph.add_edge(oid1, oid0, key=RelationshipEnum.ABOVE)
        graph.add_edge(oid0, oid1, key=RelationshipEnum.BELOW)

    if zmax1 <= zmin0:
        graph.add_edge(oid0, oid1, key=RelationshipEnum.BEHIND)
        graph.add_edge(oid1, oid0, key=RelationshipEnum.FRONT)

    if zmax0 <= zmin1:
        graph.add_edge(oid1, oid0, key=RelationshipEnum.BEHIND)
        graph.add_edge(oid0, oid1, key=RelationshipEnum.FRONT)


def add_soft_spatial_relationship_edges(
    graph: nx.MultiDiGraph,
    agent: Dict[str, Any],
    obj0: Dict[str, Any],
    obj1: Dict[str, Any],
):
    oid0 = obj0["objectId"]
    oid1 = obj1["objectId"]
    ed0 = graph.get_edge_data("agent", oid0, key=RelationshipEnum.SEES)
    ed1 = graph.get_edge_data("agent", oid0, key=RelationshipEnum.SEES)

    vis_weight = (ed0["weight"] + ed1["weight"]) / 2.0

    corners0, corners1 = get_object_bbox_corners_from_agents_ref_frame(
        agent=agent,
        obj0=obj0,
        obj1=obj1,
    )

    xmin0, ymin0, zmin0 = corners0.min(0)
    xmax0, ymax0, zmax0 = corners0.max(0)

    xmin1, ymin1, zmin1 = corners1.min(0)
    xmax1, ymax1, zmax1 = corners1.max(0)

    def compute_weight(smaller: float, larger: float):
        w = (vis_weight + 2) / 3.0
        if vis_weight > 1e-3:
            return w
        w -= (smaller <= larger) / 3.0
        w -= (
            1.0
            - sigmoid_margin_distance(
                max(smaller - larger, 0),
                margin=0,
                scale=SPATIAL_SIGMOID_SCALE,
            )
        ) / 3.0
        return w

    weight = compute_weight(smaller=xmax1, larger=xmin0)
    graph.add_edge(oid0, oid1, key=RelationshipEnum.RIGHT, weight=weight)
    graph.add_edge(oid1, oid0, key=RelationshipEnum.LEFT, weight=weight)

    weight = compute_weight(smaller=xmax0, larger=xmin1)
    graph.add_edge(oid1, oid0, key=RelationshipEnum.RIGHT, weight=weight)
    graph.add_edge(oid0, oid1, key=RelationshipEnum.LEFT, weight=weight)

    weight = compute_weight(smaller=ymax1, larger=ymin0)
    graph.add_edge(oid0, oid1, key=RelationshipEnum.ABOVE, weight=weight)
    graph.add_edge(oid1, oid0, key=RelationshipEnum.BELOW, weight=weight)

    weight = compute_weight(smaller=ymax0, larger=ymin1)
    graph.add_edge(oid1, oid0, key=RelationshipEnum.ABOVE, weight=weight)
    graph.add_edge(oid0, oid1, key=RelationshipEnum.BELOW, weight=weight)

    weight = compute_weight(smaller=zmax1, larger=zmin0)
    graph.add_edge(oid0, oid1, key=RelationshipEnum.BEHIND, weight=weight)
    graph.add_edge(oid1, oid0, key=RelationshipEnum.FRONT, weight=weight)

    weight = compute_weight(smaller=zmax0, larger=zmin1)
    graph.add_edge(oid1, oid0, key=RelationshipEnum.BEHIND, weight=weight)
    graph.add_edge(oid0, oid1, key=RelationshipEnum.FRONT, weight=weight)

    return graph


def add_agent_to_graph(graph: nx.Graph, agent: Dict[str, Any]):
    graph.add_node(
        node_for_adding="agent",
        type=NodeTypeEnum.AGENT,
        position=agent["position"],
        rotation=agent["rotation"],
    )


def process_salient_properties(salient_colors: Optional[List[str]]) -> str:
    return " and ".join(salient_colors) if salient_colors else "None"


def add_objects_to_graph(graph: nx.Graph, objects: List[Dict[str, Any]]):
    for o in objects:
        graph.add_node(
            node_for_adding=o["objectId"],
            type=NodeTypeEnum.OBJECT,
            object_type=o["objectType"],
            color=process_salient_properties(o["salientColors"]),
            material=process_salient_properties(o["salientMaterials"]),
            position=o["position"],
            rotation=o["rotation"],
            attributes=get_object_attributes(o),
        )


def add_rooms_to_graph(graph: nx.Graph, rooms: List[Dict[str, Any]]):
    for room in rooms:
        floor_polygon = Polygon(
            [position_to_floor_tuple(p) for p in room["floorPolygon"]]
        )
        graph.add_node(
            node_for_adding=room["id"],
            type=NodeTypeEnum.ROOM,
            position={
                "x": floor_polygon.centroid.x,
                "y": 0,
                "z": floor_polygon.centroid.y,
            },
            room_type=room["roomType"],
            floor_polygon=floor_polygon,
        )


def build_edgefree_relationship_graph(
    metadata: Dict[str, Any],
    rooms: List[Dict[str, Any]],
    object_ids_subset: Optional[Set[str]] = None,
    remove_walls_floors_from_object_ids=True,
) -> Tuple[List[Dict[str, Any]], StrictMultiDiGraph]:
    objects = metadata["objects"]
    if object_ids_subset is not None:
        objects = [o for o in objects if o["objectId"] in object_ids_subset]
    elif remove_walls_floors_from_object_ids:
        objects = [
            o
            for o in objects
            if o["objectType"] not in ["Floor", "Wall", "wall"]
            and o["objectId"].split("|")[0] not in ["wall", "room"]
        ]

    for o in objects:
        o["radius"] = get_object_radius(o)

    graph = StrictMultiDiGraph()
    add_agent_to_graph(graph=graph, agent=metadata["agent"])
    add_objects_to_graph(graph=graph, objects=objects)
    add_rooms_to_graph(graph=graph, rooms=rooms)

    return objects, graph


def build_relationship_graph(
    controller: Controller,
    rooms: List[Dict[str, Any]],
    object_ids_subset: Optional[Set[str]] = None,
    remove_walls_floors_from_object_ids=True,
):
    md = controller.last_event.metadata

    objects, graph = build_edgefree_relationship_graph(
        metadata=md,
        rooms=rooms,
        object_ids_subset=object_ids_subset,
        remove_walls_floors_from_object_ids=remove_walls_floors_from_object_ids,
    )

    cube, flat, long = load_categories()
    # count = 0

    object_id_to_obj = {o["objectId"]: o for o in objects}

    agent = md["agent"]
    agent_point = Point(position_to_floor_tuple(agent["position"]))
    for room in rooms:
        if graph.nodes(data=True)[room["id"]]["floor_polygon"].contains(
            agent_point
        ):
            graph.add_edge(room["id"], "agent", key=RelationshipEnum.CONTAINS)

    arm = md["arm"]
    for oid in arm["heldObjects"]:
        if oid in object_id_to_obj:
            graph.add_edge("agent", oid, key=RelationshipEnum.HOLDS)
            graph.add_edge("agent", oid, key=RelationshipEnum.HOLDS)

    for oid in arm["touchedNotHeldObjects"] + arm["heldObjects"]:
        if oid in object_id_to_obj:
            graph.add_edge("agent", oid, key=RelationshipEnum.TOUCHES)

    object_id_to_on_object_id = controller.step(
        "CheckWhatObjectsOn",
        objectIds=[o["objectId"] for o in objects],
        raise_for_failure=True,
    ).metadata["actionReturn"]

    visible_objects = []
    for i, o in enumerate(objects):
        oid = o["objectId"]
        if o["visible"]:
            graph.add_edge("agent", oid, key=RelationshipEnum.SEES)
            visible_objects.append(o)

        on_object_id = object_id_to_on_object_id[oid]
        if on_object_id is not None and on_object_id in object_id_to_obj:
            graph.add_edge(oid, on_object_id, key=RelationshipEnum.ON)

        object_point = Point(position_to_floor_tuple(o["position"]))
        for room in rooms:
            if graph.nodes(data=True)[room["id"]]["floor_polygon"].contains(
                object_point
            ):
                graph.add_edge(room["id"], oid, key=RelationshipEnum.CONTAINS)

        if o["receptacle"]:
            for contained_oid in o["receptacleObjectIds"]:
                if contained_oid in object_id_to_obj:
                    graph.add_edge(
                        oid,
                        contained_oid,
                        key=RelationshipEnum.CONTAINS,
                    )

        for o1 in objects[i + 1 :]:
            add_spatial_relationship_edges(
                graph=graph, agent=agent, obj0=o, obj1=o1
            )
            oid1 = o1["objectId"]

            if o["mass"] != 0.0 and o1["mass"] != 0.0:
                if 5 >= o["mass"] > o1["mass"]:
                    graph.add_edge(oid, oid1, key=RelationshipEnum.HEAVIERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.LIGHTERTHAN)
                elif o["mass"] < o1["mass"] <= 5:
                    graph.add_edge(oid1, oid, key=RelationshipEnum.HEAVIERTHAN)
                    graph.add_edge(oid, oid1, key=RelationshipEnum.LIGHTERTHAN)

            min1, mid1, max1 = get_size(o)
            min2, mid2, max2 = get_size(o1)
            if o["objectType"] in cube and o1["objectType"] in cube:
                if min1 <= min2 and mid1 <= mid2 and max1 <= max2:
                    # count += 1
                    graph.add_edge(oid, oid1, key=RelationshipEnum.SMALLERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.LARGERTHAN)
                if min1 >= min2 and mid1 >= mid2 and max1 >= max2:
                    # count += 1
                    graph.add_edge(oid, oid1, key=RelationshipEnum.LARGERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.SMALLERTHAN)
            elif o["objectType"] in flat and o1["objectType"] in flat:
                if mid1 <= mid2 and max1 <= max2:
                    # count += 1
                    graph.add_edge(oid, oid1, key=RelationshipEnum.SMALLERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.LARGERTHAN)
                if mid1 >= mid2 and max1 >= max2:
                    # count += 1
                    graph.add_edge(oid, oid1, key=RelationshipEnum.LARGERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.SMALLERTHAN)
            elif o["objectType"] in long and o1["objectType"] in long:
                if max1 <= max2:
                    # count += 1
                    graph.add_edge(oid, oid1, key=RelationshipEnum.SHORTERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.LONGERTHAN)
                if max1 >= max2:
                    # count += 1
                    graph.add_edge(oid, oid1, key=RelationshipEnum.LONGERTHAN)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.SHORTERTHAN)

            if fast_dist_between_objects_lower_bound(o, o1) > NEAR_THRESHOLD:
                continue

            if (
                IThorEnvironment.position_dist(o["position"], o1["position"])
                <= NEAR_THRESHOLD
                or controller.step(
                    "BBoxDistance", objectId0=oid, objectId1=oid1
                ).metadata["actionReturn"]
                <= NEAR_THRESHOLD
            ):
                graph.add_edge(oid, oid1, key=RelationshipEnum.NEAR)
                graph.add_edge(oid1, oid, key=RelationshipEnum.NEAR)

                if controller.step(
                    "CheckUnobstructedPathBetweenObjectCenters",
                    objectId0=oid,
                    objectId1=oid1,
                    raise_for_failure=True,
                ).metadata["actionReturn"]["adjacent"]:
                    graph.add_edge(oid, oid1, key=RelationshipEnum.ADJACENT)
                    graph.add_edge(oid1, oid, key=RelationshipEnum.ADJACENT)

    # print(count)
    return graph


def build_soft_relationship_graph(
    controller: Controller,
    rooms: List[Dict[str, Any]],
    object_ids_subset: Optional[Set[str]] = None,
    remove_walls_floors_from_object_ids=True,
):
    md = controller.last_event.metadata

    objects, graph = build_edgefree_relationship_graph(
        metadata=md,
        rooms=rooms,
        object_ids_subset=object_ids_subset,
        remove_walls_floors_from_object_ids=remove_walls_floors_from_object_ids,
    )

    ####################################
    # Agent <-> Object relationships ###
    ####################################
    agent = md["agent"]
    arm = md["arm"]
    agent_pos = agent["position"]
    agent_rot = agent["rotation"]
    agent_floor_pos = position_to_numpy(agent_pos, with_y=False)
    hand_pos = arm["handSphereCenter"]
    held_objects_set = set(arm["heldObjects"])
    touched_not_held_objects_set = set(arm["touchedNotHeldObjects"])
    touched_objects = held_objects_set | touched_not_held_objects_set

    for o in objects:
        oid = o["objectId"]

        ###############
        ### TOUCHES ###
        ###############
        if oid in touched_objects:
            touch_weight = 0.0
        else:
            dist_lower_bound = fast_dist_to_object_lower_bound(hand_pos, o)

            far_dist = sigmoid_margin_distance(
                dist=dist_lower_bound,
                margin=TOUCH_NEAR_THRESHOLD,
                scale=TOUCH_SIGMOID_SCALE,
            )

            near_dist = 1.0
            if far_dist < 1e-4:
                close_points = controller.step(
                    "PointOnObjectsCollidersClosestToPoint",
                    objectId=oid,
                    point=hand_pos,
                    raise_for_failure=True,
                ).metadata["actionReturn"]
                assert len(close_points) > 0
                unscaled_near_dist = min(
                    IThorEnvironment.position_dist(hand_pos, cp)
                    for cp in close_points
                )
                max_unscaled_near_dist = o["radius"] + TOUCH_NEAR_THRESHOLD
                near_dist = unscaled_near_dist / max_unscaled_near_dist

            touch_weight = (1 + far_dist + near_dist) / 3.0

        graph.add_edge(
            "agent",
            oid,
            key=RelationshipEnum.TOUCHES,
            weight=touch_weight,
        )

        #############
        ### HOLDS ###
        #############
        if o["pickupable"]:
            if oid in held_objects_set:
                held_weight = 0.0
            else:
                held_weight = (1 + 2 * touch_weight) / 3
            graph.add_edge(
                "agent",
                oid,
                key=RelationshipEnum.HOLDS,
                weight=held_weight,
            )

        ###############
        ### VISIBLE ###
        ###############
        if o["visible"]:
            visible_weight = 0.0
        else:
            approx_dist_agent_object = fast_dist_to_object_lower_bound(
                agent_pos, o
            )
            obj_agent_dist = sigmoid_margin_distance(
                dist=approx_dist_agent_object,
                margin=VISIBLE_DISTANCE,
                scale=VISIBLE_SIGMOID_SCALE,
            )

            obj_floor_pos = position_to_numpy(o["position"], with_y=False)
            dir_vec = obj_floor_pos - agent_floor_pos

            rad_angle = get_angle_of_vec_counter_clockwise(dir_vec)
            angle_in_thor = float(-(180 / np.pi) * rad_angle + 90)

            if obj_agent_dist > 1e-3:
                obj_agent_rot_dist = 1
            else:
                obj_agent_rot_dist = (
                    IThorEnvironment.rotation_dist(
                        agent_rot, {"x": 0, "y": angle_in_thor, "z": 0}
                    )
                    / 180
                )

            visible_weight = (1 + obj_agent_dist + obj_agent_rot_dist) / 3
        graph.add_edge(
            "agent",
            oid,
            key=RelationshipEnum.SEES,
            weight=visible_weight,
        )

    object_id_to_on_object_id = controller.step(
        "CheckWhatObjectsOn",
        objectIds=[o["objectId"] for o in objects],
        raise_for_failure=True,
    ).metadata["actionReturn"]

    #######################################
    ### OBJECT <-> OBJECT RELATIONSHIPS ###
    #######################################
    for i, o0 in enumerate(objects):
        oid0 = o0["objectId"]
        for o1 in objects[i + 1 :]:
            oid1 = o1["objectId"]

            ###########################################
            ### FRONT BEHIND LEFT RIGHT ABOVE BELOW ###
            ###########################################
            add_soft_spatial_relationship_edges(
                graph=graph, agent=agent, obj0=o0, obj1=o1
            )

            ############
            ### NEAR ###
            ############
            lower_bound_dist = fast_dist_between_objects_lower_bound(o0, o1)

            near_dist_lower_bound = sigmoid_margin_distance(
                dist=lower_bound_dist,
                margin=NEAR_THRESHOLD,
                scale=NEAR_SIGMOID_SCALE,
            )

            near_weight = near_dist_lower_bound / 2.0
            if near_dist_lower_bound < 1e-4:
                near_dist = controller.step(
                    "BBoxDistance", objectId0=oid0, objectId1=oid1
                ).metadata["actionReturn"]
                near_weight += (
                    sigmoid_margin_distance(
                        dist=near_dist, margin=NEAR_THRESHOLD, scale=1.0
                    )
                    / 2.0
                )
            else:
                near_weight += 1 / 2.0

            graph.add_edge(
                oid0,
                oid1,
                key=RelationshipEnum.NEAR,
                weight=near_weight,
            )
            graph.add_edge(
                oid1,
                oid0,
                key=RelationshipEnum.NEAR,
                weight=near_weight,
            )

            ################
            ### ADJACENT ###
            ################
            adj_weight = near_weight / 2.0
            if near_weight < 1e-4:
                unobstructed = controller.step(
                    "CheckUnobstructedPathBetweenObjectCenters",
                    objectId0=oid0,
                    objectId1=oid1,
                    raise_for_failure=True,
                ).metadata["actionReturn"]["adjacent"]
                adj_weight += (1 - unobstructed) / 2.0
            else:
                adj_weight += 1 / 2.0

            ##########
            ### ON ###
            ##########
            a = graph.get_edge_data(oid0, oid1, key=RelationshipEnum.NEAR)[
                "weight"
            ]
            b = graph.get_edge_data(oid0, oid1, key=RelationshipEnum.ABOVE)[
                "weight"
            ]

            on_weight = (a + b + (object_id_to_on_object_id[oid0] == oid1)) / 3
            graph.add_edge(
                oid0, oid1, key=RelationshipEnum.ON, weight=on_weight
            )

            #################
            ### CONTAINS ###
            #################
            for container, contained in [(o0, o1), (o1, o0)]:
                if container["receptacle"]:
                    container_oid = container["objectId"]
                    contained_oid = contained["objectId"]

                    if contained_oid in container["receptacleObjectIds"]:
                        contained_weight = 0
                    else:
                        contained_weight = (
                            1
                            + graph.get_edge_data(
                                container_oid,
                                contained_oid,
                                key=RelationshipEnum.NEAR,
                            )["weight"]
                        ) / 2.0

                    graph.add_edge(
                        container_oid,
                        contained_oid,
                        key=RelationshipEnum.CONTAINS,
                        weight=contained_weight,
                    )
    ###########################################
    ### ROOM <-> OBJECT/AGENT RELATIONSHIPS ###
    ###########################################
    agent_point = Point(position_to_floor_tuple(agent["position"]))
    for room in rooms:
        ######################
        ### CONTAINS AGENT ###
        ######################
        floor_poly = graph.nodes(data=True)[room["id"]]["floor_polygon"]
        agent_dist_to_floor_poly = floor_poly.distance(agent_point)
        if agent_dist_to_floor_poly < 1e-8:
            contained_weight = 0
        else:
            contained_weight = (
                1
                + sigmoid_margin_distance(
                    agent_dist_to_floor_poly,
                    margin=0,
                    scale=NEAR_SIGMOID_SCALE,
                )
            ) / 2.0
        graph.add_edge(
            room["id"],
            "agent",
            key=RelationshipEnum.CONTAINS,
            weight=contained_weight,
        )

        #######################
        ### CONTAINS OBJECT ###
        #######################
        for o in objects:
            oid = o["objectId"]

            object_point = Point(position_to_floor_tuple(o["position"]))
            object_dist_to_floor_poly = floor_poly.distance(object_point)
            if object_dist_to_floor_poly < 1e-8:
                contained_weight = 0
            else:
                contained_weight = (
                    1
                    + sigmoid_margin_distance(
                        object_dist_to_floor_poly,
                        margin=0,
                        scale=NEAR_SIGMOID_SCALE,
                    )
                ) / 2.0
            graph.add_edge(
                room["id"],
                oid,
                key=RelationshipEnum.CONTAINS,
                weight=contained_weight,
            )

    return graph


def plot_relationship_graph(graph: networkx.MultiDiGraph):
    graph = copy.deepcopy(graph)

    node_x = []
    node_y = []
    node_types = []
    type_to_color = {"AGENT": "red", "OBJECT": "blue", "ROOM": "green"}
    for node, node_data in graph.nodes(data=True):
        x, y = position_to_floor_tuple(node_data["position"])
        node_x.append(x)
        node_y.append(y)

        node_types.append(str(node_data["type"]).split(".")[-1])

    node_colors = [type_to_color[t] for t in node_types]

    node_id_to_label = {
        k: v.get("object_type", v.get("room_type", k))
        for k, v in dict(graph.nodes.data()).items()
    }

    edge_x = []
    edge_y = []
    mid_edge_x = []
    mid_edge_y = []
    edge_angle = []
    edge_vecs = []
    edge_texts = []
    for u, v, relationship_enum in graph.edges(keys=True):
        x0, y0 = position_to_floor_tuple(graph.nodes[u]["position"])
        x1, y1 = position_to_floor_tuple(graph.nodes[v]["position"])
        edge_x.append(x0)
        edge_x.append(x1)
        edge_x.append(None)
        edge_y.append(y0)
        edge_y.append(y1)
        edge_y.append(None)

        vec = np.array([x1 - x0, y1 - y0])
        norm = np.linalg.norm(vec)

        if norm != 0:
            mid_edge_x.append((x0 + x1) / 2)
            mid_edge_y.append((y0 + y1) / 2)

            vec /= norm
            edge_vecs.append(vec)

            if abs(vec[0]) < 1e-3:
                edge_angle.append(90 * (1 if vec[1] > 0 else -1))
            else:
                edge_angle.append(
                    np.arctan(vec[1] / vec[0]) * (180 / float(np.pi))
                    + (180 if vec[0] < 0 else 0)
                )

            relationship = str(relationship_enum).split(".")[-1]

            edge_data = graph.get_edge_data(u, v)[relationship_enum]
            if "weight" in edge_data:
                edge_texts.append(
                    f"{relationship}({node_id_to_label[u]}, {node_id_to_label[v]}, {edge_data['weight']:.2g})"
                )
            else:
                edge_texts.append(
                    f"{relationship}({node_id_to_label[u]}, {node_id_to_label[v]})"
                )

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        mode="lines",
        hoverinfo="text",
    )
    edge_trace.text = []

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        # marker=dict(
        #     showscale=True,
        #     # colorscale options
        #     # 'Greys' | 'YlGnBu' | 'Greens' | 'YlOrRd' | 'Bluered' | 'RdBu' |
        #     # 'Reds' | 'Blues' | 'Picnic' | 'Rainbow' | 'Portland' | 'Jet' |
        #     # 'Hot' | 'Blackbody' | 'Earth' | 'Electric' | 'Viridis' |
        #     colorscale="YlGnBu",
        #     reversescale=True,
        #     color=[],
        #     size=10,
        #     colorbar=dict(
        #         thickness=15, title="Node Connections", xanchor="left", titleside="right",
        #     ),
        #     line_width=2,
        # ),
        marker=dict(
            size=10,
            line_width=2,
        ),
    )

    node_adjacencies = []
    node_text = []
    for node, adjacencies in zip(graph.nodes, graph.adjacency()):
        node_adjacencies.append(len(adjacencies[1]))
        node_text.append(node_id_to_label[node])
        if len(graph.nodes[node].get("attributes", [])) != 0:
            node_text[-1] += (
                " ("
                + ", ".join(
                    str(attr).split(".")[-1]
                    for attr in graph.nodes[node]["attributes"]
                )
                + ")"
            )

    node_trace.marker.color = node_colors
    node_trace.text = node_text

    arrow_traces = []
    for x, y, vec, edge_text in zip(
        mid_edge_x, mid_edge_y, edge_vecs, edge_texts
    ):
        orth = np.array([-vec[1], vec[0]])

        p0 = np.array([x, y]) + 0.04 * vec
        p1 = p0 - 0.03 * (vec + orth)
        p2 = p0 - 0.03 * (vec - orth)
        p3 = p0
        xs, ys = unzip([p0, p1, p2, p3], None)
        arrow_traces.append(
            go.Scatter(
                x=xs,
                y=ys,
                fill="toself",
                line=dict(color="#888", width=2),
                marker=dict(opacity=0),
                hoverinfo="text",
            )
        )
        arrow_traces[-1].text = []

    last_x, last_y = float("inf"), float("inf")
    last_edge_text = ""
    num_concat_texts = 0
    new_edge_texts = []
    new_mid_x = []
    new_mid_y = []
    for x, y, edge_text in sorted(
        list(
            zip(
                mid_edge_x + [1000],
                mid_edge_y + [1000],
                edge_texts + ["__END__"],
            )
        )
    ):
        if (
            np.linalg.norm(np.array((last_x, last_y)) - np.array((x, y)))
            < 1e-3
        ):
            if last_edge_text != "":
                if num_concat_texts > 0 and num_concat_texts % 3 == 0:
                    edge_text = f"{last_edge_text} &<br> {edge_text}"
                else:
                    edge_text = f"{last_edge_text} & {edge_text}"
                num_concat_texts += 1
        elif last_edge_text != "":
            num_concat_texts = 0
            new_edge_texts.append(last_edge_text)
            new_mid_x.append(last_x)
            new_mid_y.append(last_y)

        last_x, last_y = x, y
        last_edge_text = edge_text

    invisible_edge_node_trace = go.Scatter(
        x=new_mid_x,
        y=new_mid_y,
        text=new_edge_texts,
        mode="markers",
        hoverinfo="text",
        marker=dict(opacity=0),
    )

    fig = go.Figure(
        data=[
            edge_trace,
            node_trace,
            *arrow_traces,
            invisible_edge_node_trace,
        ],
        layout=go.Layout(
            title="Graph of ProcTHOR House",
            titlefont_size=16,
            showlegend=False,
            hovermode="closest",
            margin=dict(b=20, l=5, r=5, t=40),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
        ),
    )

    iplot(fig)


def main():
    if "houses_dict" not in locals():
        houses_dict = datasets.load_dataset(
            "allenai/houses", use_auth_token=True
        )

    if "c" not in locals():
        c = Controller(
            branch="nanna-grasp-force",
            scene="Procedural",
            agentMode="arm",
            fastActionEmit=True,
        )

    c.reset()

    train_houses = houses_dict["train"]

    # house = pickle.loads(train_houses[10]["house"])
    house = pickle.loads(train_houses[9]["house"])
    c.step("CreateHouse", house=house, raise_for_failure=True)

    c.step(
        "TeleportFull",
        **house["metadata"]["agent"],
        raise_for_failure=True,
    )

    for _ in range(6):
        c.step("MoveAhead")

    metadata = c.step("AdvancePhysicsStep", simSeconds=2).metadata

    graph = build_relationship_graph(
        controller=c,
        object_ids_subset=set(
            o["objectId"]
            for o in metadata["objects"]
            if o["objectType"] not in ["Floor", "Wall", "wall"]
            and o["objectId"].split("|")[0] not in ["wall", "room"]
        ),
        rooms=house["rooms"],
    )
    # pickle.dump(graph, open("graph.pkl", "wb"))

    plot_relationship_graph(graph)


if __name__ == "__main__":
    main()
