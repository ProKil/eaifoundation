import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
from jax import jit, random


def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = (
            value.numpy()
        )  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _float_feature(value):
    """Returns a float_list from a float / double."""
    return tf.train.Feature(float_list=tf.train.FloatList(value=value))


@jit
def split_and_sample(key):
    key, subkey = random.split(key)
    val = random.normal(subkey, shape=(128, 512))
    return key, val


key, val = split_and_sample(random.PRNGKey(0))

questions = [
    "Is there an apple in the kitchen?",
    "Is there a chair close to the lamp?",
    "Wher is the bathroom?",
]

answers = [
    "Yes",
    "No",
    "Behind the couch",
]

dataset = [
    {"memory": val[i], "inputs": questions[i % 3], "targets": answers[i % 3]}
    for i in range(128)
]


def serialize_example(
    memory: jax.numpy.DeviceArray, inputs: str, targets: str
):
    """
    Creates a tf.train.Example message ready to be written to a file.
    """
    # Create a dictionary mapping the feature name to the tf.train.Example-compatible
    # data type.
    feature = {
        "memory": _float_feature(np.array(memory, copy=False)),
        "inputs": _bytes_feature(inputs.encode("utf-8")),
        "targets": _bytes_feature(targets.encode("utf-8")),
    }

    # Create a Features message using tf.train.Example.

    example_proto = tf.train.Example(
        features=tf.train.Features(feature=feature)
    )
    return example_proto.SerializeToString()


with tf.io.TFRecordWriter("probing_qa.tfrecord") as writer:
    for data in dataset:
        serialized = serialize_example(**data)
        writer.write(serialized)
