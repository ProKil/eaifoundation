from collections import Counter

import numpy as np
import numpy.typing as npt

relationships: npt.NDArray = np.load("uvk_types.npy", allow_pickle=True)
relationship_counter = Counter([tuple(i) for i in relationships.tolist()])
relationships_sorted_by_frequency = sorted(
    relationship_counter.items(), key=lambda x: x[1], reverse=True
)
with open("relationships_sorted_by_frequency.txt", "w") as f:
    for relationship, count in relationships_sorted_by_frequency:
        f.write(f"{relationship} {count}\n")
