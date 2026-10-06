import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from src.quality import BLUR_THRESHOLD, inspect_image


def encode(img: np.ndarray) -> bytes:
    ok, buf = cv2.imencode(".png", img)
    assert ok
    return bytes(buf)


def sharp_fixture() -> np.ndarray:
    rng = np.random.default_rng(11)
    base = np.full((320, 320, 3), 128, dtype=np.uint8)
    base[::16, :] = 255
    base[:, ::16] = 0
    noise = rng.integers(0, 30, (320, 320, 3), dtype=np.uint8)
    return cv2.add(base, noise)


def test_sharp_image_accepted():
    report = inspect_image(encode(sharp_fixture()))
    assert report.is_acceptable is True
    assert report.blur_score >= BLUR_THRESHOLD
    assert report.rejection_reason is None


def test_blurred_image_rejected():
    sharp = sharp_fixture()
    blurred = cv2.GaussianBlur(sharp, (31, 31), 0)
    report = inspect_image(encode(blurred))
    assert report.is_acceptable is False
    assert report.rejection_reason == "image_blurred"


def test_dark_image_rejected():
    dark = np.full((320, 320, 3), 5, dtype=np.uint8)
    report = inspect_image(encode(dark))
    assert report.is_acceptable is False
    assert report.rejection_reason == "image_underexposed"


def test_bright_image_rejected():
    bright = np.full((320, 320, 3), 250, dtype=np.uint8)
    report = inspect_image(encode(bright))
    assert report.is_acceptable is False
    assert report.rejection_reason == "image_overexposed"


def test_too_small_rejected():
    tiny = np.full((64, 64, 3), 128, dtype=np.uint8)
    report = inspect_image(encode(tiny))
    assert report.is_acceptable is False
    assert report.rejection_reason == "image_too_small"


def test_undecodable_rejected():
    report = inspect_image(b"not-an-image")
    assert report.is_acceptable is False
    assert report.rejection_reason == "image_undecodable"
