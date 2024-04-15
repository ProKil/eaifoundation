from pickletools import optimize

from flax.traverse_util import ModelParamTraversal

from t5x.adafactor import Adafactor
from t5x.optimizers import MultiOptimizer


def PromptOptimizer():
    prompt_generator_params = ModelParamTraversal(
        lambda path, _: "prompt" in path
    )
    opt = Adafactor()
    return MultiOptimizer([(prompt_generator_params, opt)])
