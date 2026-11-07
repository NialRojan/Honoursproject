"""Frozen, leak-free cross-validation splits.

Texts that share a page, or that normalise to the same template ("Only 3 left" /
"only 17 LEFT"), are linked into one group and never straddle train/test.
"""
import re

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def normalise_template(text: str) -> str:
    t = str(text).lower()
    t = re.sub(r"\d+", "#", t)
    t = re.sub(r"[^\w#\s]", "", t)
    return " ".join(t.split())


def make_groups(df) -> np.ndarray:
    """Union-find over rows that share a page_id or a normalised template."""
    n = len(df)
    parent = list(range(n))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    keys = [df["page_id"].tolist(), [normalise_template(t) for t in df["text"]]]
    for col in keys:
        first = {}
        for i, k in enumerate(col):
            if k in first:
                parent[find(i)] = find(first[k])
            else:
                first[k] = i
    return np.array([find(i) for i in range(n)])


def grouped_folds(y, groups, k=5, seed=42):
    sgkf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    return [(tr, te) for tr, te in sgkf.split(idx, y, groups)]


def random_folds(y, k=5, seed=42):
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=seed)
    idx = np.arange(len(y))
    return [(tr, te) for tr, te in skf.split(idx, y)]
