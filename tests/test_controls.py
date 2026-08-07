import numpy as np

from glys_rul.controls import shuffled_label_control


def test_shuffling_preserves_the_label_multiset():
    rng = np.random.default_rng(0)
    target = np.array([3.0, 24.0, 47.0, 73.0, 78.0, 100.0])
    groups = np.array(["a", "b", "c", "d", "e", "f"])

    shuffled = shuffled_label_control(target, groups, rng)

    assert sorted(shuffled) == sorted(target)


def test_shuffling_preserves_the_multiset_with_duplicate_groups():
    """Real groups hold two differing labels, so the permutation works per sample."""
    rng = np.random.default_rng(0)
    target = np.array([3.0, 5.0, 47.0, 51.0, 100.0])
    groups = np.array(["a", "a", "b", "b", "c"])

    shuffled = shuffled_label_control(target, groups, rng)

    assert sorted(shuffled) == sorted(target)


def test_shuffling_actually_changes_the_assignment():
    rng = np.random.default_rng(3)
    target = np.arange(10, 110, 10, dtype=float)
    groups = np.array([f"g{index}" for index in range(10)])

    shuffled = shuffled_label_control(target, groups, rng)

    assert not np.array_equal(shuffled, target)


def test_control_is_reproducible_for_a_given_seed():
    target = np.arange(10, 110, 10, dtype=float)
    groups = np.array([f"g{index}" for index in range(10)])

    first = shuffled_label_control(target, groups, np.random.default_rng(5))
    second = shuffled_label_control(target, groups, np.random.default_rng(5))

    assert np.array_equal(first, second)


def test_different_seeds_give_different_permutations():
    target = np.arange(10, 110, 10, dtype=float)
    groups = np.array([f"g{index}" for index in range(10)])

    first = shuffled_label_control(target, groups, np.random.default_rng(5))
    second = shuffled_label_control(target, groups, np.random.default_rng(6))

    assert not np.array_equal(first, second)


def test_group_structure_is_left_untouched():
    """Only labels are permuted; the grouping that drives the splits must survive."""
    rng = np.random.default_rng(0)
    target = np.array([3.0, 5.0, 47.0, 51.0, 100.0])
    groups = np.array(["a", "a", "b", "b", "c"])
    original = groups.copy()

    shuffled_label_control(target, groups, rng)

    assert np.array_equal(groups, original)


def test_correlation_with_the_original_labels_is_destroyed():
    """Averaged over many draws the permuted labels carry no signal."""
    target = np.arange(1, 21, dtype=float)
    groups = np.array([f"g{index}" for index in range(20)])

    correlations = [
        np.corrcoef(target, shuffled_label_control(target, groups, np.random.default_rng(seed)))[
            0, 1
        ]
        for seed in range(50)
    ]

    assert abs(float(np.mean(correlations))) < 0.2
