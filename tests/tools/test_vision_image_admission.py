"""Raster pixels must decode before native attachment or auxiliary inference."""

import base64
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from PIL import Image

from tools import vision_tools as vt


# Synthetic RGB header with RGBA scanlines, valid PNG chunk CRCs and zlib data.
# It reproduces the admission defect without retaining a customer's logo.
MALFORMED = Path(__file__).parents[1] / "fixtures/images/rgb_header_rgba_scanlines.png"


def _image_url(result):
    return next(part["image_url"]["url"] for part in result["content"]
                if part.get("type") == "image_url")


async def _analyze(route, source):
    if route == "native":
        return await vt._vision_analyze_native(source, "Describe", task_id="image-admission")
    return await vt.vision_analyze_tool(source, "Describe", task_id="image-admission")


def test_fixture_passes_header_verification_but_cannot_decode():
    with Image.open(MALFORMED) as image:
        assert image.size == (180, 180)
        image.verify()
    with Image.open(MALFORMED) as image, pytest.raises(OSError):
        image.load()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "auxiliary"])
@pytest.mark.parametrize("source_kind", ["path", "file", "data"])
async def test_bad_raster_returns_recoverable_error_before_encoding(route, source_kind, tmp_path):
    source = tmp_path / "bad.png"
    source.write_bytes(MALFORMED.read_bytes())
    image_url = str(source)
    if source_kind == "file":
        image_url = source.as_uri()
    elif source_kind == "data":
        image_url = "data:image/png;base64," + base64.b64encode(source.read_bytes()).decode()
    with patch.object(vt, "_image_to_base64_data_url") as encode, \
         patch.object(vt, "async_call_llm", new_callable=AsyncMock) as provider:
        result = json.loads(await _analyze(route, image_url))
    assert result["success"] is False
    assert "fully decoded" in result["error"]
    assert "retry vision_analyze" in result["error"]
    assert "_multimodal" not in result
    encode.assert_not_called()
    provider.assert_not_awaited()
    assert source.read_bytes() == MALFORMED.read_bytes()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "auxiliary"])
async def test_rejection_cleans_temp_copy_but_preserves_source(route, tmp_path):
    source = tmp_path / "bad.png"
    source.write_bytes(MALFORMED.read_bytes())
    cache = tmp_path / "cache"
    with patch.object(vt, "get_hermes_dir", return_value=cache):
        result = json.loads(await _analyze(route, str(source)))
    assert result["success"] is False
    assert list(cache.iterdir()) == []
    assert source.read_bytes() == MALFORMED.read_bytes()


def test_single_image_over_pillow_hard_pixel_bound_is_rejected(tmp_path, monkeypatch):
    source = tmp_path / "large.png"
    Image.new("RGB", (20, 20)).save(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 80)
    path, mime, error = vt._normalize_to_supported_image(source, "image/png")
    assert path is None and mime is None
    assert "pixels" in error


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "auxiliary"])
@pytest.mark.parametrize("image_format", ["PNG", "JPEG", "GIF", "WEBP"])
async def test_valid_supported_image_keeps_bytes_after_bad_image(route, image_format, tmp_path):
    source = tmp_path / "valid.img"
    Image.new("RGB", (8, 8), "red").save(source, format=image_format)
    original = source.read_bytes()
    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A red square"))])
    with patch.object(vt, "async_call_llm", new_callable=AsyncMock, return_value=response) as provider:
        rejected = json.loads(await _analyze(route, str(MALFORMED)))
        assert rejected["success"] is False
        provider.assert_not_awaited()
        accepted = await _analyze(route, str(source))
    if route == "native":
        assert accepted["_multimodal"] is True
        data_url = _image_url(accepted)
        provider.assert_not_awaited()
    else:
        assert json.loads(accepted)["success"] is True
        provider.assert_awaited_once()
        data_url = _image_url(provider.await_args.kwargs["messages"][0])
    assert base64.b64decode(data_url.partition(",")[2]) == original
    assert source.read_bytes() == original


@pytest.mark.parametrize("image_format,mime", [("BMP", "image/bmp"), ("TIFF", "image/tiff")])
def test_valid_raster_conversion_remains_decodable(image_format, mime, tmp_path):
    source = tmp_path / "source.img"
    Image.new("RGB", (8, 8), "blue").save(source, format=image_format)
    original = source.read_bytes()
    path, normalized_mime, error = vt._normalize_to_supported_image(source, mime)
    assert error is None
    assert normalized_mime == "image/png"
    with Image.open(path) as image:
        image.load()
        assert image.getpixel((0, 0)) == (0, 0, 255)
    assert source.read_bytes() == original
    path.unlink()


@pytest.mark.parametrize("malformed", [False, True])
def test_svg_output_is_decoded_and_failed_output_is_removed(malformed, tmp_path):
    source = tmp_path / "source.svg"
    source.write_text('<svg xmlns="http://www.w3.org/2000/svg"/>', encoding="utf-8")
    converted = []

    def rasterize(_source, output):
        converted.append(output)
        if malformed:
            output.write_bytes(MALFORMED.read_bytes())
        else:
            Image.new("RGB", (8, 8), "blue").save(output, format="PNG")
        return True

    with patch.object(vt, "_rasterize_svg_to_png", side_effect=rasterize):
        path, mime, error = vt._normalize_to_supported_image(source, "image/svg+xml")
    assert source.exists()
    if malformed:
        assert path is None and mime is None
        assert "fully decoded" in error
        assert not converted[0].exists()
    else:
        assert error is None and mime == "image/png"
        with Image.open(path) as image:
            image.load()
        path.unlink()


def _animated_gif(path):
    frames = [Image.new("RGB", (8, 8), color) for color in ("red", "blue", "green")]
    frames[0].save(path, format="GIF", save_all=True, append_images=frames[1:])


def test_valid_animation_keeps_every_frame_and_original_bytes(tmp_path):
    source = tmp_path / "animation.gif"
    _animated_gif(source)
    original = source.read_bytes()
    path, mime, error = vt._normalize_to_supported_image(source, "image/gif")
    assert (path, mime, error) == (source, "image/gif", None)
    assert path.read_bytes() == original
    with Image.open(path) as image:
        assert image.n_frames == 3


def test_corrupt_later_animation_frame_is_rejected(tmp_path):
    source = tmp_path / "animation.gif"
    _animated_gif(source)
    original = source.read_bytes()
    # Leave a decodable first frame but truncate the last frame's compressed data.
    source.write_bytes(original[:-8])
    with Image.open(source) as image:
        image.load()
    path, mime, error = vt._normalize_to_supported_image(source, "image/gif")
    assert path is None and mime is None
    assert "fully decoded" in error


def test_animation_decode_budget_covers_all_frames(tmp_path, monkeypatch):
    source = tmp_path / "animation.gif"
    _animated_gif(source)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 80)
    # Every frame is 64 pixels; the third exceeds the shared 160-pixel budget.
    path, mime, error = vt._normalize_to_supported_image(source, "image/gif")
    assert path is None and mime is None
    assert "decode pixel limit" in error


def test_missing_decoder_does_not_admit_raster(tmp_path):
    source = tmp_path / "valid.png"
    Image.new("RGB", (8, 8)).save(source)
    with patch.dict("sys.modules", {"PIL.ImageSequence": None}), \
         patch("builtins.__import__", side_effect=ImportError("Pillow unavailable")):
        path, mime, error = vt._normalize_to_supported_image(source, "image/png")
    assert path is None and mime is None
    assert "Pillow unavailable" in error
    assert source.exists()


@pytest.mark.asyncio
@pytest.mark.parametrize("route", ["native", "auxiliary"])
async def test_validation_uses_bounded_cpu_executor(route, tmp_path):
    import threading

    source = tmp_path / "valid.png"
    Image.new("RGB", (8, 8)).save(source)
    threads = []
    decode = vt._decode_image_for_vision

    def observe(path):
        threads.append(threading.current_thread().name)
        decode(path)

    response = SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="A square"))])
    with patch.object(vt, "_decode_image_for_vision", side_effect=observe), \
         patch.object(vt, "async_call_llm", new_callable=AsyncMock, return_value=response):
        await _analyze(route, str(source))
    assert threads and all(name.startswith("vision-encode") for name in threads)
