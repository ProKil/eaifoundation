import functools

from t5x.state_utils import apply_assignment_map


def MemoryQARestoreFromT5StateTransformationFns():
    return (
        functools.partial(
            apply_assignment_map,
            assignment_map=[
                (r"state.*", None),
                (r"target/prompt_generator.*", None),
            ],
        ),
    )
