import numpy as np

from glys_rul.dataset import augment_batch, grouped_splits


def test_no_group_appears_in_both_sides_of_a_split():
    groups = np.array(["a", "a", "b", "b", "c"])

    for train, test in grouped_splits(groups):
        assert not set(groups[train]) & set(groups[test])


def test_leave_one_group_out_yields_one_fold_per_group():
    groups = np.array(["a", "a", "b", "b", "c"])

    folds = list(grouped_splits(groups))

    assert len(folds) == 3


def test_every_sample_is_tested_exactly_once():
    groups = np.array(["a", "a", "b", "b", "c"])

    tested = np.concatenate([test for _, test in grouped_splits(groups)])

    assert sorted(tested) == list(range(len(groups)))


def test_duplicate_samples_are_held_out_together():
    """Byte-identical images must never be split across train and test."""
    groups = np.array(["a", "a", "b", "b", "c"])

    for _, test in grouped_splits(groups):
        held = set(groups[test])
        for label in held:
            assert (groups[test] == label).sum() == (groups == label).sum()


def test_many_groups_fall_back_to_k_fold():
    groups = np.array([f"g{index}" for index in range(25)])

    folds = list(grouped_splits(groups, max_logo_groups=10, k=5))

    assert len(folds) == 5
    for train, test in folds:
        assert not set(groups[train]) & set(groups[test])


def test_augmentation_returns_the_same_number_of_samples():
    rng = np.random.default_rng(0)
    images = rng.random((4, 32, 32, 1))

    augmented = augment_batch(images, rng)

    assert augmented.shape == images.shape


def test_augmentation_preserves_the_temperature_range():
    """Photometric jitter is forbidden: colour encodes the label."""
    rng = np.random.default_rng(0)
    images = np.full((4, 32, 32, 1), 0.5)

    augmented = augment_batch(images, rng, noise_std=0.0, calibration_bias_std=0.0)

    interior = augmented[:, 8:24, 8:24, :]
    assert np.allclose(interior, 0.5, atol=1e-6), "geometry-only augmentation must not shift values"


def test_augmentation_is_deterministic_for_a_given_seed():
    images = np.random.default_rng(1).random((3, 16, 16, 1))

    first = augment_batch(images, np.random.default_rng(7))
    second = augment_batch(images, np.random.default_rng(7))

    assert np.array_equal(first, second)


def test_different_seeds_produce_different_augmentations():
    images = np.random.default_rng(1).random((3, 16, 16, 1))

    first = augment_batch(images, np.random.default_rng(7))
    second = augment_batch(images, np.random.default_rng(8))

    assert not np.array_equal(first, second)


def test_sensor_noise_is_applied_when_requested():
    rng = np.random.default_rng(0)
    images = np.full((4, 16, 16, 1), 0.5)

    augmented = augment_batch(
        images, rng, noise_std=0.01, max_shift=0, max_scale=0.0, max_rotation_deg=0.0,
        calibration_bias_std=0.0,
    )

    assert not np.allclose(augmented, 0.5)
    assert np.abs(augmented - 0.5).max() < 0.1
