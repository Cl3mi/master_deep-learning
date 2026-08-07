import numpy as np
from PIL import Image

from glys_rul.io import load_rgb, md5_of


def test_transparent_pixels_become_white_not_black(tmp_path):
    img = Image.new("RGBA", (2, 1), (255, 0, 0, 255))
    img.putpixel((1, 0), (0, 0, 0, 0))  # fully transparent
    path = tmp_path / "alpha.png"
    img.save(path)

    out = load_rgb(path)

    assert np.allclose(out[0, 0], [255, 0, 0])
    assert np.allclose(out[0, 1], [255, 255, 255]), "transparent pixel must composite to white"


def test_semi_transparent_pixel_blends_towards_white(tmp_path):
    img = Image.new("RGBA", (1, 1), (0, 0, 0, 128))
    path = tmp_path / "semi.png"
    img.save(path)

    out = load_rgb(path)

    assert 120 < out[0, 0, 0] < 135


def test_opaque_rgb_image_is_unchanged(tmp_path):
    path = tmp_path / "plain.jpeg"
    Image.new("RGB", (4, 4), (10, 20, 30)).save(path, quality=100)

    out = load_rgb(path)

    assert out.shape == (4, 4, 3)
    assert out.dtype == np.float64


def test_md5_is_stable_and_distinguishes_content(tmp_path):
    a, b = tmp_path / "a.bin", tmp_path / "b.bin"
    a.write_bytes(b"hello")
    b.write_bytes(b"world")

    assert md5_of(a) == md5_of(a)
    assert md5_of(a) != md5_of(b)
