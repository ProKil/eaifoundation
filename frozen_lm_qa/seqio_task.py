"""Register the memory probe qa task.

Example use:
    >>> import seqio
    >>> dataset = seqio.get_mixture_or_task('memory_probe_qa').get_dataset(
    >>>     sequence_length={"question": 256, "answer": 128},
    >>>     split="train",
    >>>     shuffle=True,
    >>>     num_epochs=1,
    >>>     shard_info=seqio.ShardInfo(index=0, num_shards=10),
    >>>     use_cached=False,
    >>>     seed=42
    >>> )
    >>>
    >>> # Print the first 5 examples.
    >>> for _, ex in zip(range(5), dataset.as_numpy_iterator()):
    >>>     print(ex)
"""


from copy import deepcopy
from functools import partial
from typing import Mapping

import seqio
import tensorflow as tf
from t5.data import get_default_vocabulary


def tokenize_question_and_answer(
    dataset: tf.data.Dataset,
    output_features: seqio.preprocessors.OutputFeaturesType,
    copy_pretokenized: bool = True,
    with_eos: bool = False,
):
    _output_features = deepcopy(output_features)
    _output_features.pop("memory")  # memory should not be tokenized
    return seqio.preprocessors.tokenize(
        dataset=dataset,
        output_features=_output_features,
        copy_pretokenized=copy_pretokenized,
        with_eos=with_eos,
    )


seqio.TaskRegistry.add(
    name="memory_probe_qa",
    source=seqio.TFExampleDataSource(
        {
            "train": "gs://haoz-bucket/data/probing_qa_dataset/probing_qa_train_[0-7]*.tfrecord",
            "validation": "gs://haoz-bucket/data/probing_qa_dataset/probing_qa_train_8*.tfrecord",
            "test": "gs://haoz-bucket/data/probing_qa_dataset/probing_qa_train_9*.tfrecord",
        },
        feature_description={
            "memory": tf.io.FixedLenFeature([512], tf.float32),
            "inputs": tf.io.FixedLenFeature([], tf.string),
            "targets": tf.io.FixedLenFeature([], tf.string),
        },
    ),
    preprocessors=[
        tokenize_question_and_answer,
        seqio.preprocessors.append_eos,
    ],
    output_features={
        "memory": seqio.ContinuousFeature(dtype=tf.float32, add_eos=False),
        "inputs": seqio.Feature(
            get_default_vocabulary(),
            add_eos=True,
        ),
        "targets": seqio.Feature(
            get_default_vocabulary(),
            add_eos=True,
        ),
    },
)
