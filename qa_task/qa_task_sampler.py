import pickle
import warnings
from typing import Any, Dict, Optional, Sequence, Union

import canonicaljson
from allenact.base_abstractions.task import Sensor
from allenact.utils.misc_utils import md5_hash_str_as_int
from question_answering.question_generation import question_generation
from question_answering.questions.questions import realize_question
from tokenizers import Tokenizer

from boolset.tasks_and_samplers import (
    AgentPose,
    HouseAugmenter,
    ProcTHORDataset,
    Vector3,
)
from qa_task.procthor_task import ProcTHORTask
from qa_task.procthor_task_sampler import ProcTHORTaskSampler
from qa_task.qa_task import QATask


class QATaskSampler(ProcTHORTaskSampler):
    def __init__(
        self,
        procthor_dataset: ProcTHORDataset,
        sensors: Sequence[Sensor],
        house_repeats: Union[int, float],
        max_steps: int,
        reset_on_scene_replay: bool,
        augmenter: Optional[HouseAugmenter] = None,
        controller_kwargs: Optional[Dict[str, Any]] = None,
        seed: Optional[int] = None,
        allow_house_skip: bool = True,
        num_qa_pairs: int = 1,
        max_answer_length: int = 32,
        uvk_pickle_file: Optional[str] = None,
        tokenizer: Optional[Tokenizer] = None,
        **kwargs,
    ):
        super(QATaskSampler, self).__init__(
            procthor_dataset,
            sensors,
            house_repeats,
            max_steps,
            reset_on_scene_replay,
            augmenter,
            controller_kwargs,
            seed,
            allow_house_skip,
            **kwargs,
        )
        assert uvk_pickle_file is not None, "uvk_pickle_file is required!"
        assert tokenizer is not None, "tokenizer is required!"
        self.num_qa_pairs = num_qa_pairs
        self.max_answer_length = max_answer_length
        self.tokenizer = tokenizer
        self.uvk = pickle.load(open(uvk_pickle_file, "rb"))

    def next_task(self, force_advance_scene: bool = False) -> Optional[QATask]:
        if (
            force_advance_scene
            or self._last_sampled_task is None  # type: ignore
            or self._current_repeat_ind >= self.house_repeats
        ):
            self._setup_next_house(controller=self.controller)
        else:
            if self.reset_on_scene_replay:
                self.controller.reset()
                self.procthor_dataset.initialize_house(
                    controller=self.controller, index=self._current_house_ind
                )
            else:
                self.controller.step(
                    action="ResetObjectFilter",
                    raise_for_failure=True,
                )

        self._current_repeat_ind += 1

        # Randomize agent pose
        n_retries = 10
        for i in range(n_retries):
            standing = (
                {}
                if self.controller.initialization_parameters["agentMode"]
                == "locobot"
                else {"standing": self._random.choice([False, True])}
            )
            starting_pose = AgentPose(  # type: ignore
                position=self._random.choice(
                    self.procthor_dataset.reachable_positions_map[
                        self._current_house_ind
                    ]
                ),
                rotation=Vector3(x=0, y=self._random.random() * 360, z=0),
                horizon=self._random.randint(-1, 2) * 30,
                **standing,
            )
            md = self.controller.step(
                action="TeleportFull", **starting_pose
            ).metadata
            if not md["lastActionSuccess"]:
                if i == n_retries - 1:
                    warnings.warn(
                        f"Teleport failed in {self._current_house_ind} {n_retries} times!"
                    )
                continue
            break
        self.augmenter.apply(self.controller)

        graph = self.procthor_dataset.relationships_graph_map[
            self._current_house_ind
        ]

        questions, answers = [], []
        for i in range(self.num_qa_pairs):
            for _ in range(10):
                qa = question_generation(graph, lambda: self.uvk)
                if qa is not None:
                    question, answer = qa
                    question_str = realize_question(question)
                    answer_str = "Yes" if answer else "No"
                    questions.append(question_str)
                    answers.append(answer_str)
                    break
            else:
                raise Exception(
                    f"No question generated for house {self._current_house_ind}!"
                )

        self._last_sampled_task = QATask(
            env=self.controller,
            sensors=self.sensors,
            max_steps=self.max_steps,
            house=self._current_house,
            task_info={
                "questions": questions,
                "answers": answers,
                "house_md5": md5_hash_str_as_int(
                    canonicaljson.encode_canonical_json(
                        self._current_house
                    ).decode("utf-8")
                ),
                "max_answer_length": self.max_answer_length,
            },
            tokenizer=self.tokenizer,
        )
        return self._last_sampled_task
