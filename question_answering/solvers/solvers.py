from functools import singledispatch
from itertools import chain

from question_answering.questions import QuestionTemplate
from question_answering.questions.questions import (
    CountingQuestion,
    ExistenceQuestion,
    RelationshipQuestion,
)
from relationship_graph.graphs import StrictMultiDiGraph

from .solver_elements import *


class NotAnswerableError(Exception):
    pass


@singledispatch
def solve(question: QuestionTemplate, graph: StrictMultiDiGraph):
    raise NotAnswerableError(f"No solver for question type {type(question)}")


@solve.register(CountingQuestion)
def _(question: CountingQuestion, graph: StrictMultiDiGraph) -> int:
    room_type: str = question.obj_ids[0]
    object_type: str = question.obj_ids[1]
    return len(
        list(
            chain(
                *map(
                    partial(
                        room2objects,
                        object_type=object_type,
                        graph=graph,
                    ),
                    filter(
                        partial(
                            find_room_w_type,
                            room_type=room_type,
                            graph=graph,
                        ),
                        graph.nodes(),
                    ),
                )
            )
        )
    )


@solve.register(ExistenceQuestion)
def _(question: ExistenceQuestion, graph: StrictMultiDiGraph) -> bool:
    room_type: str = question.obj_ids[0]
    object_type: str = question.obj_ids[1]
    return not is_empty(
        chain(
            *map(
                partial(room2objects, object_type=object_type, graph=graph),
                filter(
                    partial(
                        find_room_w_type,
                        room_type=room_type,
                        graph=graph,
                    ),
                    graph.nodes(),
                ),
            )
        )
    )


@solve.register(RelationshipQuestion)
def _(question: RelationshipQuestion, graph: StrictMultiDiGraph) -> bool:
    rel_type: RelationshipEnum = question.relation_type
    object1_type: str = question.obj_ids[0]
    object2_type: str = question.obj_ids[1]
    return not is_empty(
        chain(
            *map(
                partial(filter, lambda vk: vk == (object2_type, rel_type)),
                map(
                    partial(get_out_neighbors_type_w_relation, graph=graph),
                    filter(
                        partial(
                            find_object_w_type,
                            object_type=object1_type,
                            graph=graph,
                        ),
                        graph.nodes(),
                    ),
                ),
            )
        )
    )
