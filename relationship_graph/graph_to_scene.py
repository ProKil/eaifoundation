import json
import pickle
from functools import partial
from typing import Dict, Set

import datasets
import tqdm
from ai2thor.controller import Controller
from question_answering.solvers.solver_elements import (
    edge2uvktype,
    get_node_color,
    get_node_material,
    get_node_type,
)
from relationship_graph.graphs import (
    StrictMultiDiGraph,
    build_relationship_graph,
)

import prior
from boolset.tasks_and_samplers import ProcTHORDataset


def graph_to_scene(graph: StrictMultiDiGraph) -> Dict:
    object2idx = {obj: idx for idx, obj in enumerate(graph.nodes)}
    result = dict()
    result["objects"] = [
        dict(
            objecttype=get_node_type(graph.nodes[i]),
            color=get_node_color(graph.nodes[i]),
            material=get_node_material(graph.nodes[i]),
        )
        for i in graph.nodes()
    ]
    result["relationships"] = {}
    for i in graph.edges:
        relation_name = str(i[2]).split(".")[-1].lower()
        if relation_name not in result["relationships"]:
            result["relationships"][relation_name] = [
                [] for _ in range(len(object2idx))
            ]
        result["relationships"][relation_name][object2idx[i[1]]].append(
            object2idx[i[0]]
        )
    return result


def composing_scenes():
    scenes = list()
    for i in range(0, 10):
        scene = json.load(
            open(f"procthor-10k-scenes-{i*1000}-{(i+1)*1000}.json", "r")
        )
        scenes.extend(scene["scenes"])
    scenes = dict(info=dict(split="train"), scenes=scenes)
    return scenes


if __name__ == "__main__":
    if "houses_dict" not in locals():
        houses_dict = prior.load_dataset("procthor-10k")

    if "c" not in locals():
        c = Controller(
            branch="nanna-grasp-force",
            scene="Procedural",
            agentMode="arm",
            fastActionEmit=True,
            platform="CloudRendering",
        )

    c.reset()

    train_houses = houses_dict["train"]

    scenes = dict(info=dict(split="train"), scenes=[])

    dataset = ProcTHORDataset([houses_dict["train"][i] for i in range(10000)])

    for i in tqdm.tqdm(range(0, 10000)):
        try:
            dataset.initialize_house(c, i)

            scene = graph_to_scene(dataset.relationships_graph_map[i])
            scenes["scenes"].append(scene)
        except:
            scenes["scenes"].append({"objects": [], "relationships": {}})

        if (i + 1) % 1000 == 0:
            json.dump(
                scenes,
                open(f"procthor-10k-scenes-{i-999}-{i+1}.json", "w"),
            )
            scenes = dict(info=dict(split="train"), scenes=[])
