from dataclasses import dataclass
from typing import List, Tuple

import numpy as np
from relationship_graph.attrs_relations import RelationshipEnum


@dataclass
class QuestionTemplate(object):
    obj_ids: Tuple[str, str]
    nl_form: str

    @classmethod
    def from_triplet(
        cls, triplet: Tuple[str, str, RelationshipEnum]
    ) -> "QuestionTemplate":
        relationship: RelationshipEnum = RelationshipEnum(triplet[2])
        if relationship == RelationshipEnum.CONTAINS:
            return ExistenceQuestion(obj_ids=triplet[:2])
        elif relationship == RelationshipEnum.ON:
            return OnRelationshipQuestion(obj_ids=triplet[:2])
        elif relationship == RelationshipEnum.ADJACENT:
            return AdjacentRelationshipQuestion(obj_ids=triplet[:2])
        elif relationship == RelationshipEnum.NEAR:
            return NearRelationshipQuestion(obj_ids=triplet[:2])
        else:
            raise NotImplementedError(f"{relationship} is not implemented")


class CountingQuestion(QuestionTemplate):
    nl_form: str = "How many {1}s are in {0}?"


@dataclass
class ExistenceQuestion(QuestionTemplate):
    nl_form: str = "Is there a {1} in {0}?"


class RelationshipQuestion(QuestionTemplate):
    @property
    def relation_type(self) -> RelationshipEnum:
        raise NotImplementedError


@dataclass
class OnRelationshipQuestion(RelationshipQuestion):
    nl_form: str = "Is there a {0} on {1}?"

    @property
    def relation_type(self) -> RelationshipEnum:
        return RelationshipEnum.ON


@dataclass
class AdjacentRelationshipQuestion(RelationshipQuestion):
    nl_form: str = "Is there a {0} close to {1}?"

    @property
    def relation_type(self) -> RelationshipEnum:
        return RelationshipEnum.ADJACENT


@dataclass
class NearRelationshipQuestion(RelationshipQuestion):
    nl_form: str = "Is there a {0} near {1}?"

    @property
    def relation_type(self) -> RelationshipEnum:
        return RelationshipEnum.NEAR


def realize_question(question: QuestionTemplate) -> str:
    return question.nl_form.format(*question.obj_ids)
