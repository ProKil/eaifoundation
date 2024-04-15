from asyncio.format_helpers import extract_stack
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import gym
import gym.spaces
import torch
import transformers
from allenact.algorithms.onpolicy_sync.policy import (
    ActorCriticModel,
    LinearActorHead,
    LinearCriticHead,
)
from allenact.base_abstractions.misc import (
    ActorCriticOutput,
    Memory,
    ObservationType,
)
from allenact.embodiedai.models.basic_models import RNNStateEncoder
from gym.spaces.dict import Dict as SpaceDict
from torch import nn
from transformers import T5Tokenizer

# for loading the flax checkpoint
from frozen_lm_qa.flax_model import FlaxT5ForMemoryQuestionAnswering
from frozen_lm_qa.pytorch_model import T5ForMemoryQuestionAnswering
from qa_task.distributions import DeltaDistr, JointDistr

setattr(
    transformers,
    "FlaxT5ForMemoryQuestionAnswering",
    FlaxT5ForMemoryQuestionAnswering,
)

import pdb
import sys


class ForkedPdb(pdb.Pdb):
    """A Pdb subclass that may be used
    from a forked multiprocessing child
    """

    def interaction(self, *args, **kwargs):
        _stdin = sys.stdin
        try:
            sys.stdin = open("/dev/stdin")
            pdb.Pdb.interaction(self, *args, **kwargs)
        finally:
            sys.stdin = _stdin


class Unsqueeze(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor):
        return x.unsqueeze(self.dim)


class Permute(nn.Module):
    def __init__(self, permutation: Tuple[int, ...]):
        super().__init__()
        self.permutation = permutation

    def forward(self, x: torch.Tensor):
        return x.permute(self.permutation)


class FlattenNext(nn.Module):
    def __init__(self, start_dim: int):
        super().__init__()
        self.start_dim = start_dim

    def forward(self, x: torch.Tensor):
        shape = list(x.shape)
        shape = (
            *shape[: self.start_dim],
            shape[self.start_dim] * shape[self.start_dim + 1],
            *shape[self.start_dim + 2 :],
        )
        return x.view(shape)


class EmbodiedQAActorCritic(ActorCriticModel[JointDistr]):
    def __init__(
        self,
        action_space: gym.spaces.Tuple,
        observation_space: SpaceDict,
        obs_sensor_uuids: Sequence[str],
        t5_model_path: str,  # path to t5 config file, required
        hidden_size=512,
        auxiliary_uuids: Optional[List[str]] = None,
    ):
        super().__init__(action_space, observation_space)
        self._hidden_size = hidden_size
        self.auxiliary_uuids = (
            auxiliary_uuids if auxiliary_uuids is not None else []
        )

        uuid_to_input_encoder: Dict[str, nn.Module] = {}
        self.uuid_to_num_inputs = {}
        self.obs_sensor_uuids = obs_sensor_uuids
        for uuid in obs_sensor_uuids:
            obs_space: gym.Space = observation_space[uuid]
            assert isinstance(
                obs_space, (gym.spaces.Box, gym.spaces.MultiDiscrete)
            )
            if len(obs_space.shape) == 1:
                # Assumes vector input
                uuid_to_input_encoder[uuid] = nn.Sequential(
                    Unsqueeze(2),
                    nn.Linear(obs_space.shape[0], hidden_size, bias=False),
                )
                self.uuid_to_num_inputs[uuid] = 1
            elif len(obs_space.shape) == 2:
                # Assumes features are along the -1 axis
                uuid_to_input_encoder[uuid] = nn.Linear(
                    obs_space.shape[-1], hidden_size, bias=False
                )
                self.uuid_to_num_inputs[uuid] = obs_space.shape[0]
            elif len(obs_space.shape) == 3:
                # Assumes CNN shaped input so that channels are first
                uuid_to_input_encoder[uuid] = nn.Sequential(
                    FlattenNext(start_dim=3),
                    Permute((0, 1, 3, 2)),
                    nn.Linear(obs_space.shape[0], hidden_size, bias=False),
                )
                self.uuid_to_num_inputs[uuid] = (
                    obs_space.shape[1] * obs_space.shape[2]
                )
            else:
                raise NotImplementedError

        self.uuid_to_obs_sensor_encoder = nn.ModuleDict(uuid_to_input_encoder)
        self.prev_action_embedder = nn.Embedding(
            self.action_space[0].n + 1, hidden_size
        )
        self.state_encoder = RNNStateEncoder(
            (sum(self.uuid_to_num_inputs.values()) + 1) * hidden_size,
            self._hidden_size,
            num_layers=1,
            rnn_type="GRU",
            trainable_masked_hidden_state=False,
        )
        # self.create_qa_head(t5_model_path)
        self.create_actorcritic_head()

    def _recurrent_memory_specification(self):
        return {
            "working_memory": (
                (
                    ("sampler", None),
                    ("memory_emb", self._hidden_size),
                ),
                torch.float32,
            )
        }

    def forward_encoder(self, observations: ObservationType) -> torch.Tensor:
        encodings = []
        for uuid in self.obs_sensor_uuids:
            encodings.append(
                self.uuid_to_obs_sensor_encoder[uuid](observations[uuid])
            )

        return torch.cat(encodings, dim=2)

    def create_actorcritic_head(self):
        self.actor = LinearActorHead(self._hidden_size, self.action_space[0].n)
        self.critic = LinearCriticHead(self._hidden_size)

    def create_qa_head(self, t5_model_path: str):
        self.lm_qa_head = T5ForMemoryQuestionAnswering.from_pretrained(
            t5_model_path, from_flax=True
        )
        for name, param in self.lm_qa_head.named_parameters():
            if not "prompt" in name:
                param.requires_grad = False
        self.prepend_attention_mask = nn.Parameter(
            torch.ones((1, self.lm_qa_head.config.n_prompt_tokens)).long(),
            requires_grad=False,
        )
        self.t5_tokenizer = T5Tokenizer.from_pretrained("t5-base")

    def forward_qa_head(
        self,
        memory_input: torch.Tensor,
        input_ids: torch.LongTensor,
        input_attention_mask: torch.LongTensor,
        labels: Optional[torch.LongTensor] = None,
    ) -> torch.Tensor:
        """This forward supports both training and inference.
        For training, labels should be provided for the loss to be computed.
        For inference, labels should be None.
        """
        attention_mask = torch.cat(
            [
                self.prepend_attention_mask.expand(
                    *input_attention_mask.shape[:-1], -1
                ),
                input_attention_mask,
            ],
            dim=-1,
        )
        if labels is not None:
            self.lm_qa_head.gradient_checkpointing_enable()
            return self.lm_qa_head(
                memory_input=torch.flatten(
                    memory_input.expand(-1, -1, input_ids.size(2), -1), 0, 2
                ),
                input_ids=torch.flatten(input_ids, 0, 2),
                attention_mask=torch.flatten(attention_mask, 0, 2),
                labels=torch.flatten(labels, 0, 2),
                use_cache=False,
            ).loss
        else:
            self.lm_qa_head.gradient_checkpointing_disable()
            with torch.no_grad():
                results = self.lm_qa_head.generate(
                    memory_input=torch.flatten(
                        memory_input.expand(-1, -1, input_ids.size(2), -1),
                        0,
                        2,
                    ),
                    input_ids=torch.flatten(input_ids, 0, 2),
                    attention_mask=torch.flatten(attention_mask, 0, 2),
                    max_length=32,
                )
                results = torch.cat(
                    [
                        results,
                        torch.zeros(
                            *results.shape[:-1], 32 - results.shape[-1]
                        )
                        .long()
                        .to(results.device),
                    ],
                    dim=-1,
                )
                return results

    def forward(
        self,
        observations: ObservationType,
        memory: Memory,
        prev_actions: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[ActorCriticOutput[JointDistr], Optional[Memory]]:
        obs_embeds = self.forward_encoder(observations)
        if isinstance(prev_actions, torch.Tensor):
            physical_actions = prev_actions
        else:
            physical_actions, _ = prev_actions
        steps, samplers = physical_actions.size()
        questions_input_ids = observations["aux_observations_output"][
            "questions_input_ids"
        ]
        questions_input_attention_mask = observations[
            "aux_observations_output"
        ]["questions_attention_mask"]
        answers_input_ids = observations["aux_observations_output"][
            "answers_input_ids"
        ]
        prev_actions_embeds = self.prev_action_embedder(
            torch.where(
                condition=masks.view(*physical_actions.shape).byte(),
                input=physical_actions + 1,
                other=torch.zeros_like(physical_actions),
            )
        ).unsqueeze(2)
        joint_embeds = torch.cat([obs_embeds, prev_actions_embeds], dim=2)

        outputs, working_memory = self.state_encoder(
            joint_embeds,
            memory.tensor("working_memory").unsqueeze(0),  # layer
            masks=masks,
        )
        memory.set_tensor("working_memory", working_memory.squeeze(0))

        # aux_loss from qa
        # loss = self.forward_qa_head(
        #     memory_input=outputs,
        #     input_ids=questions_input_ids,
        #     input_attention_mask=questions_input_attention_mask,
        #     labels=answers_input_ids,
        # )
        loss = 0

        # prediction from qa
        # predictions = self.forward_qa_head(
        #     memory_input=outputs,
        #     input_ids=questions_input_ids,
        #     input_attention_mask=questions_input_attention_mask,
        # )
        predictions = answers_input_ids
        flattened_predictions = predictions.view(steps, samplers, -1)

        extras = {"aux_loss": loss}

        actor_critic_output = ActorCriticOutput(
            distributions=JointDistr(
                (
                    self.actor(outputs.view(steps, samplers, -1)),
                    DeltaDistr(flattened_predictions),
                )
            ),
            values=self.critic(outputs.view(steps, samplers, -1)),
            extras=extras,
        )

        return actor_critic_output, memory
