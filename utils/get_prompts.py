import random
import numpy as np


def get_prompts(path, num_per_class=1):
    idx = []
    captions = []
    f = open(path, "r").readlines()
    for c in f:
        for _ in range(num_per_class):
            captions.append(c.strip())
            idx.append(c.strip().replace(" ", "_"))
    index=random.sample(range(1,len(captions)+1),1500)
    captions=[captions[i] for i in index]
    idx=[idx[i] for i in index]
    return idx, captions
