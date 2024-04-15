import functools
import os
import pickle
import signal
import traceback
from collections import defaultdict
from typing import Any, Iterator, Mapping

import networkx as nx
import ray
import torch
import tqdm
from ai2thor.controller import Controller
from datasets import DatasetDict, load_dataset
from relationship_graph.graphs import build_relationship_graph


def fix_object_names(house):
    known_assets = defaultdict(int)
    to_traverse = house["objects"][:]
    while len(to_traverse):
        cur_obj = to_traverse.pop()
        cur_obj[
            "id"
        ] = f'{cur_obj["assetId"]}_{known_assets[cur_obj["assetId"]]}'
        known_assets[cur_obj["assetId"]] += 1
        if "children" in cur_obj:
            to_traverse.extend(cur_obj["children"][:])
    return house


def scene_folders_iterator(dir: str) -> Iterator[str]:
    for scene_folder in os.listdir(dir):
        if os.path.isdir(os.path.join(dir, scene_folder)):
            if scene_folder.startswith("train"):
                yield scene_folder


def trajectory_iterator(scene_folder: str) -> Iterator[str]:
    for traj_file in os.listdir(
        os.path.join(
            os.environ["TRAJECTORY_DIR_PATH"],
            "data",
            "expert_trajectories",
            scene_folder,
        )
    ):
        if traj_file.endswith(".pt"):
            yield traj_file


class TimedOutExc(Exception):
    """Raised when a timeout happens."""


def timeout(timeout):
    """Return a decorator that raises a TimedOutExc exception after timeout
    seconds, if the decorated function did not return."""

    def decorate(f):
        def handler(signum, frame):
            raise TimedOutExc()

        @functools.wraps(f)  # Preserves the documentation, name, etc.
        def new_f(*args, **kwargs):

            old_handler = signal.signal(signal.SIGALRM, handler)
            signal.alarm(timeout)

            result = f(*args, **kwargs)  # f() always returns, in this scheme

            signal.signal(
                signal.SIGALRM, old_handler
            )  # Old signal handler is restored
            signal.alarm(0)  # Alarm removed

            return result

        return new_f

    return decorate


def build_graph(scene_folder: str, houses_dict: Mapping) -> None:
    try:
        c = None

        @timeout(600)
        def inner():
            c = Controller(
                branch="nanna-bboxdist",
                scene="Procedural",
                agentMode="arm",
                fastActionEmit=True,
                gpu_device=hash(scene_folder) % 2,
            )

            train_houses = houses_dict["train"]
            house = fix_object_names(
                pickle.loads(
                    train_houses[int(scene_folder.split("_")[-1])]["house"]
                )
            )

            c.reset()
            c.step("CreateHouse", house=house, raise_for_failure=True)
            try:
                c.step(
                    "TeleportFull",
                    **house["metadata"]["agent"],
                    raise_for_failure=True,
                )
            except:
                print(f"Failed to teleport {scene_folder}")
                c.stop()
                return

            for _ in range(6):
                c.step("MoveAhead")

            metadata = c.step("AdvancePhysicsStep", simSeconds=2).metadata

            for traj_index, traj_file in enumerate(
                trajectory_iterator(scene_folder)
            ):
                traj_metadata_path = os.path.join(
                    os.environ["TRAJECTORY_DIR_PATH"],
                    "data",
                    "expert_trajectories",
                    f"{scene_folder}",
                    f"{traj_file}",
                )
                traj_metadata = torch.load(traj_metadata_path)
                visible_object_list, _ = traj_metadata
                objectid2obj = {o["objectId"]: o for o in metadata["objects"]}
                objectname2obj = {o["name"]: o for o in metadata["objects"]}
                graph_per_time_step = [
                    build_relationship_graph(
                        controller=c,
                        object_ids_subset=set(
                            (
                                objectid2obj[objname]
                                if objname in objectid2obj
                                else objectname2obj[objname]
                            )["objectId"]
                            for objname in visible_objects
                            if objname != "agent"
                        ),
                        rooms=house["rooms"],
                    )
                    for visible_objects in visible_object_list
                ]
                cumulative_graph = [None] * len(graph_per_time_step)
                cumulative_graph[0] = graph_per_time_step[0]
                for i in range(1, len(graph_per_time_step)):
                    cumulative_graph[i] = nx.compose(
                        cumulative_graph[i - 1], graph_per_time_step[i]
                    )

                torch.save(
                    (graph_per_time_step, cumulative_graph),
                    traj_metadata_path.replace(".pt", ".graph.pt"),
                )
            c.stop()

        inner()
    except Exception as e:
        print(
            f"Failed to build graph for {scene_folder} with error {type(e).__name__}: {traceback.format_exc()}"
        )
        if type(c) is Controller:
            c.stop()


def to_iterator(obj_ids) -> Iterator[Any]:
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


def main():
    assert "TRAJECTORY_DIR_PATH" in os.environ, "Must set TRAJECTORY_DIR_PATH"
    build_graph_remote = ray.remote(num_cpus=2)(build_graph).remote
    houses_dict = load_dataset(
        "allenai/houses",
        revision="ithor-splits",
        use_auth_token=True,
        cache_dir=os.path.expanduser("~/.new_cache"),
    )

    futures = []

    for scene_folder in tqdm.tqdm(
        scene_folders_iterator(
            os.path.join(
                os.environ["TRAJECTORY_DIR_PATH"],
                "data",
                "expert_trajectories",
            )
        )
    ):
        futures.append(build_graph_remote(scene_folder, houses_dict))

    for i in tqdm.tqdm(to_iterator(futures), total=len(futures)):
        pass


def debug_scene_folder(scene_folder: str):
    assert "TRAJECTORY_DIR_PATH" in os.environ, "Must set TRAJECTORY_DIR_PATH"
    houses_dict: DatasetDict = load_dataset("allenai/houses", revision="ithor-splits", use_auth_token=True, cache_dir=os.path.expanduser("~/.new_cache"))  # type: ignore

    build_graph(scene_folder, houses_dict)


if __name__ == "__main__":
    if os.environ["DEBUG_SCENE_FOLDER"]:
        debug_scene_folder(os.environ["DEBUG_SCENE_FOLDER"])
    else:
        main()
