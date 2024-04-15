import logging
import pickle
from functools import partial

import datasets
import numpy as np
import ray
import timeout_decorator
import tqdm
from ai2thor.controller import Controller
from question_answering.solvers.solver_elements import edge2uvktype
from relationship_graph.graphs import build_relationship_graph
from tqdm.contrib.logging import logging_redirect_tqdm

LOG = logging.getLogger(__name__)


@ray.remote
def get_uvk_triples(start: int, end: int):
    houses_dict = datasets.load_dataset("allenai/houses", use_auth_token=True)

    c = Controller(
        branch="nanna-bboxdist",
        scene="Procedural",
        agentMode="arm",
        fastActionEmit=True,
    )

    train_houses = houses_dict["train"]

    unique_uvk_types_list = []

    for idx in range(start, end):
        i = train_houses[idx]
        # load house config from training set
        house = pickle.loads(i["house"])

        @timeout_decorator.timeout(30)
        def get_graph():
            # controller setup
            c.reset()
            c.step("CreateHouse", house=house, raise_for_failure=True)
            try:
                c.step(
                    "TeleportFull",
                    **house["metadata"]["agent"],
                    raise_for_failure=True,
                )
            except RuntimeError as e:
                LOG.warning(f"Failed to teleport agent to house {idx}: {e}")
                raise RuntimeError(e)
            # for _ in range(6):
            #     c.step("MoveAhead")

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

            return graph

        try:
            graph = get_graph()
        except timeout_decorator.TimeoutError:
            print(f"Timeout skipping {idx}")
            continue
        except RuntimeError:
            continue

        unique_uvk_types = list(
            set(map(partial(edge2uvktype, graph=graph), graph.edges))
        )
        unique_uvk_types_list.append(np.array(unique_uvk_types, dtype=object))

    c.stop()
    uvk_types = np.vstack(unique_uvk_types_list)
    return uvk_types


def to_iterator(obj_ids):
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


if __name__ == "__main__":
    ray.init()
    logging.basicConfig(level=logging.INFO)
    with logging_redirect_tqdm():
        job_list = [
            (get_uvk_triples.remote(i * 100, (i + 1) * 100))
            for i in range(100)
        ]
        uvk_types = np.vstack(
            list(tqdm.tqdm(to_iterator(job_list), total=len(job_list)))
        )
    np.save("uvk_types_1.npy", uvk_types)
