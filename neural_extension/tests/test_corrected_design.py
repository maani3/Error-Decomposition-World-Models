"""Structural tests for the corrected paired experiment."""

import os
import sys

import numpy as np

BASE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, BASE)

from run_corrected import trajectory_ids, trajectory_split  # noqa: E402


def test_trajectory_split_and_nested_prefixes():
    ids = trajectory_ids(500, 30)
    train_group, val_group = trajectory_split(ids)
    assert not np.any(train_group & val_group)
    assert np.all(train_group | val_group)

    previous_train = set()
    previous_val = set()
    expected_counts = {5: (4, 1), 15: (13, 2), 40: (36, 4),
                       150: (135, 15), 500: (450, 50)}
    for n0 in (5, 15, 40, 150, 500):
        prefix = ids < n0
        train_ids = set(np.unique(ids[prefix & train_group]).tolist())
        val_ids = set(np.unique(ids[prefix & val_group]).tolist())
        assert previous_train.issubset(train_ids)
        assert previous_val.issubset(val_ids)
        assert train_ids.isdisjoint(val_ids)
        assert (len(train_ids), len(val_ids)) == expected_counts[n0]
        previous_train, previous_val = train_ids, val_ids


if __name__ == "__main__":
    test_trajectory_split_and_nested_prefixes()
    print("corrected trajectory split and nested-prefix checks passed")

