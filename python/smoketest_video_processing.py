from __future__ import annotations

import numpy as np

import golfcar_backend.video_processing as vp


def main() -> None:
    print(vp.check_env())

    handler = vp.ImageHandler()
    assert handler.interpolation_flags == vp.InterpolationFlags.LINEAR
    assert tuple(handler.pad_color)[:3] == (114.0, 114.0, 114.0)

    handler.interpolation_flags = vp.InterpolationFlags.NEAREST
    handler.pad_color = (1, 2, 3)
    assert handler.interpolation_flags == vp.InterpolationFlags.NEAREST
    assert tuple(handler.pad_color)[:3] == (1.0, 2.0, 3.0)

    image = np.zeros((20, 40, 3), dtype=np.uint8)
    result = handler.letterbox(image, 64)
    assert isinstance(result, vp.LetterBoxResult)
    assert result.image.shape == (64, 64, 3)
    assert result.image.dtype == np.uint8
    assert result.pad_x == 0
    assert result.pad_y == 16
    assert abs(result.scale - 1.6) < 1e-5
    assert result.to_dict()["image"].shape == (64, 64, 3)

    reverted = handler.letterbox_revert(result)
    assert reverted.shape == image.shape
    assert reverted.dtype == np.uint8

    gray = np.zeros((10, 15), dtype=np.uint8)
    assert handler.letterbox(gray, 32).image.shape == (32, 32)

    rgba = np.zeros((10, 15, 4), dtype=np.uint8)
    assert handler.letterbox(rgba, 32).image.shape == (32, 32, 4)

    print("VIDEO_PROCESSING_SMOKETEST_OK")


if __name__ == "__main__":
    main()
