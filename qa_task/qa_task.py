"""Question answering task for ProcTHOR environment.
The difference between this task and the original ProcTHOR task is that this task
provides questions and answers as an observation,
which are not expected to be used by the model, but rather by the auxiliary loss.
The model should provide aux_model to the auxiliary loss.
"""

import numbers
import re
from typing import Any, Dict, List, Tuple, Union

import gym
import numpy.typing as npt
import torch
from allenact.base_abstractions.misc import (
    ObservationType,
    RLStepResult,
)
from gym import spaces
from tokenizers import Tokenizer

from qa_task.embodied_qa_model import ForkedPdb
from qa_task.procthor_task import ProcTHORTask


class QATask(ProcTHORTask):
    def __init__(self, tokenizer: Tokenizer, *args, **kwargs):
        super(QATask, self).__init__(*args, **kwargs)
        self.tokenizer = tokenizer
        self.questions_str: List[str] = self.task_info["questions"]
        self.answers_str: List[str] = self.task_info["answers"]
        self.questions = tokenizer(
            self.questions_str,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
        )
        self.answers = tokenizer(
            self.answers_str,
            return_tensors="pt",
            padding="max_length",
            max_length=32,
        )
        assert len(self.questions) == len(
            self.answers
        ), "Number of questions and answers must be equal."
        self.max_answer_length: int = self.task_info["max_answer_length"]
        self.vocab_size: int = self.tokenizer.vocab_size
        self.last_accuracy: float = 0.0

    @property
    def action_space(self) -> gym.Space:
        return spaces.Tuple(
            [
                spaces.Discrete(len(self.ACTIONS)),
                spaces.MultiDiscrete(
                    [self.vocab_size]
                    * len(self.questions)
                    * self.max_answer_length
                ),
            ]
        )

    def get_observations(self, **kwargs) -> ObservationType:
        """This function overrides the original function in the allenact.base_abstraction.Task class.
        The conventional treatment is add observations as sensors, but here we directly add them as part
        of the output of observations.
        """
        result: ObservationType = super(QATask, self).get_observations(
            **kwargs
        )
        result["aux_observations"] = dict(
            questions_input_ids=self.questions["input_ids"],
            answers_input_ids=self.answers["input_ids"],
            questions_attention_mask=self.questions["attention_mask"],
        )
        return result

    def get_accuracy_improvement(self, answers: List[str]) -> float:
        accuracy: float = sum(
            map(lambda x: x[0] == x[1], zip(answers, self.answers_str))
        ) / len(self.answers_str)

        accuracy, self.last_accuracy = self.last_accuracy, accuracy
        return self.last_accuracy - accuracy

    def metrics(self) -> Dict[str, Any]:
        return {
            **super(QATask, self).metrics(),
            "accuracy": self.last_accuracy,
        }

    def _step(self, action: Tuple[int, List[int]]) -> RLStepResult:
        physical_action, answers_list = action
        answers = torch.Tensor(answers_list).long()
        answers = answers.view(len(self.questions_str), -1)
        result = super(QATask, self)._step(physical_action)
        answers_str: List[str] = self.tokenizer.batch_decode(
            answers, skip_special_tokens=True
        )
        result = result.clone(
            {
                "reward": result.reward
                + self.get_accuracy_improvement(answers_str)
            }
        )  # add qa rewards
        return result
