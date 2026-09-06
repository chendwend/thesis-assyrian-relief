from __future__ import annotations

from PIL import Image
import pytest

from thesis_assyrian_relief.utils.data import ResizeAndPad, build_resize_transform


def test_resize_and_pad_preserves_content_aspect_ratio() -> None:
    image = Image.new("RGB", (400, 100), color=(255, 255, 255))

    result = ResizeAndPad(size=224)(image)

    assert result.size == (224, 224)
    assert result.getpixel((112, 84)) == (255, 255, 255)
    assert result.getpixel((112, 0)) == (124, 116, 104)


def test_resize_transform_rejects_unknown_mode() -> None:
    with pytest.raises(ValueError, match="Unknown resize mode"):
        build_resize_transform("crop")
