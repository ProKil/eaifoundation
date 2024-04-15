import pickle
from functools import partial
from typing import Callable, Optional, Set, Tuple, Union

import numpy as np
from question_answering.questions.questions import QuestionTemplate
from question_answering.solvers.solver_elements import (
    edge2uvktype,
    get_node_type,
)
from question_answering.solvers.solvers import solve
from relationship_graph.graphs import StrictMultiDiGraph


def get_all_uvk(
    get_high_freq_relationship: Callable,
    u: Optional[Set[str]] = None,
    v: Optional[Set[str]] = None,
):
    return list(
        filter(
            lambda x: (u is None or x[0] in u) and (v is None or x[1] in v),
            get_high_freq_relationship(),
        )
    )


def question_generation(
    graph: StrictMultiDiGraph,
    get_high_freq_relationship: Callable,
    from_fix_set_rate=0.5,
) -> Optional[Tuple[QuestionTemplate, Union[int, bool]]]:
    objects: Set[str] = set(
        get_node_type(graph.nodes[i]) for i in graph.nodes()
    )
    triplets = get_all_uvk(
        get_high_freq_relationship, u=objects
    ) + get_all_uvk(get_high_freq_relationship, v=objects)
    true_triplets = list(
        set(map(partial(edge2uvktype, graph=graph), graph.edges))
    )
    if len(triplets) == 0:
        return None
    sample_triplet = (
        triplets[np.random.randint(len(triplets))]
        if np.random.random() < from_fix_set_rate
        else true_triplets[np.random.randint(len(true_triplets))]
    )
    try:
        question = QuestionTemplate.from_triplet(sample_triplet)
    except NotImplementedError:
        return None
    answer = solve(question, graph)
    return question, answer
