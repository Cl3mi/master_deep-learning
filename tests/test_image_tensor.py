import numpy as np
import pytest

from glys_rul.colorscale import ColorScale
from glys_rul.dataset import build_image_tensor
from glys_rul.evaluate import build_feature_table

pytestmark = pytest.mark.slow

TINY = {"min_component_px": 100, "erode_iterations": 1}


def test_tensor_has_one_normalised_map_per_sample(engine_dir, scale_image):
    table = build_feature_table(engine_dir, scale_image, **TINY)
    scale = ColorScale.from_image(scale_image)

    tensor = build_image_tensor(table, scale, engine_dir, image_size=32)

    assert tensor.shape == (len(table), 32, 32, 1)
    assert tensor.min() >= 0.0 and tensor.max() <= 1.0


def test_hotter_engine_yields_a_higher_mean_map(engine_dir, scale_image):
    table = build_feature_table(engine_dir, scale_image, **TINY)
    scale = ColorScale.from_image(scale_image)

    tensor = build_image_tensor(table, scale, engine_dir, image_size=32)
    hottest = int(table["rul"].idxmin())
    coldest = int(table["rul"].idxmax())

    assert tensor[hottest].mean() > tensor[coldest].mean()


def test_tensor_is_deterministic(engine_dir, scale_image):
    table = build_feature_table(engine_dir, scale_image, **TINY)
    scale = ColorScale.from_image(scale_image)

    first = build_image_tensor(table, scale, engine_dir, image_size=32)
    second = build_image_tensor(table, scale, engine_dir, image_size=32)

    assert np.array_equal(first, second)


def test_duplicate_files_produce_identical_tensors(engine_dir, scale_image):
    """Byte-identical images must yield identical inputs, or grouping is meaningless."""
    table = build_feature_table(engine_dir, scale_image, **TINY)
    scale = ColorScale.from_image(scale_image)

    tensor = build_image_tensor(table, scale, engine_dir, image_size=32)
    for _, members in table.groupby("group"):
        indices = list(members.index)
        for other in indices[1:]:
            assert np.array_equal(tensor[indices[0]], tensor[other])


def test_row_order_follows_the_feature_table(engine_dir, scale_image):
    """Tensor row i must correspond to table row i, or labels are misaligned."""
    table = build_feature_table(engine_dir, scale_image, **TINY)
    scale = ColorScale.from_image(scale_image)

    tensor = build_image_tensor(table, scale, engine_dir, image_size=32)
    means = tensor.mean(axis=(1, 2, 3))

    assert np.corrcoef(means, table["total_c"].to_numpy())[0, 1] > 0.9
