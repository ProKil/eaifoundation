import pathlib
import pickle
from functools import lru_cache, partial
from typing import Any, Iterator, List, Optional, Set, Tuple, Union

import h5py
import numpy as np
import ray
import tensorflow as tf
import torch
import tqdm
from question_answering.question_generation import question_generation
from question_answering.questions import (
    QuestionTemplate,
    realize_question,
)
from question_answering.solvers import solve
from question_answering.solvers.solver_elements import (
    edge2uvktype,
    get_node_type,
)
from relationship_graph.graphs import StrictMultiDiGraph

from frozen_lm_qa.build_dataset import serialize_example

uvk = pickle.load(open("high_frequency_relationship.pkl", "rb"))


def get_all_scenes(
    dir="/net/nfs2.prior/hao/procthor_trajectories/data/trajectories_with_memory/",
):
    return pathlib.Path(dir).glob("train_*")


def get_all_trajectories(dir):
    return pathlib.Path(dir).glob("*.graph.pt")


def get_question_answers(trajectory_path: pathlib.Path):
    graphs = torch.load(trajectory_path)
    memory_path = str(trajectory_path).replace(".graph.pt", ".hdf5")
    memory_file = h5py.File(memory_path, "r")
    dataset = memory_file["memory"]["rnn"]
    memory_qa = list()
    for i, j in zip(graphs[1], dataset):  # graphs[1] are the cumulative graphs
        for _ in range(10):
            qa = question_generation(i, lambda: uvk)
            if qa is not None:
                question, answer = qa
                question_str = realize_question(question)
                answer_str = "Yes" if answer else "No"
                memory = np.array(j).reshape(-1)
                memory_qa.append(
                    dict(
                        memory=memory, inputs=question_str, targets=answer_str
                    )
                )
                break
        else:
            raise Exception("No question generated")
    return memory_qa


@ray.remote
def get_data_for_scene(scene: pathlib.Path):
    trajectories = get_all_trajectories(scene)
    memory_qa = list()
    for trajectory in trajectories:
        memory_qa.extend(get_question_answers(trajectory))
    with tf.io.TFRecordWriter(
        f"/net/nfs2.prior/hao/probing_qa_dataset/probing_qa_{scene.stem}.tfrecord"
    ) as writer:
        for qa in memory_qa:
            writer.write(serialize_example(**qa))


def to_iterator(obj_ids) -> Iterator[Any]:
    while obj_ids:
        done, obj_ids = ray.wait(obj_ids)
        yield ray.get(done[0])


if __name__ == "__main__":
    ray.init()
    all_scenes = list(get_all_scenes())
    jobs = []
    for scene in all_scenes:
        jobs.append(get_data_for_scene.remote(scene))

    for i in tqdm.tqdm(to_iterator(jobs), total=len(jobs)):
        pass
